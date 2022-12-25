from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from datasets import load_dataset
import torchvision.transforms as transform
import torch
from torchvision.transforms import InterpolationMode 
import numpy as np
import torch.utils.data as baseData
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import train
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


train_dataset = load_dataset("Chris1/cityscapes", split="train")
val_dataset = load_dataset("Chris1/cityscapes", split="validation")


mean=torch.tensor([0.4850, 0.4560, 0.4060])
std=torch.tensor([0.2290, 0.2240, 0.2250])


transformsImage = transform.Compose(
    [
        transform.Resize((512,512),interpolation=InterpolationMode.BILINEAR),
        transform.ToTensor(),
        transform.Normalize(mean, std)
    ]
)

transformsLabel = transform.Compose(
    [
        transform.Resize((512,512),interpolation=InterpolationMode.NEAREST),
        transform.Lambda(lambda t: torch.as_tensor(np.array(t), dtype=torch.int64))
    ]
)

postprocess = transform.Compose([
     transform.Lambda(lambda t: (t.cpu() * std.reshape(3,1,1))+mean.reshape(3,1,1)),
     transform.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     transform.Lambda(lambda t: t * 255.),
     transform.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
])


import torch.utils.data as baseData
import torch
from torch.utils.data import DataLoader

class CityscapeData(baseData.Dataset):
    def __init__(self,ds,img_sz=512):
        super(CityscapeData, self).__init__()
        self.img_sz=img_sz
        self.ds=ds
        self.class_values=[i for i in range(35)]
        
    def __getitem__(self, index):
        
        img = self.ds[index]["image"]
        masks = self.ds[index]["semantic_segmentation"]
        img=transformsImage(img)
        masks=transformsLabel(masks)[...,0]
        #masks = [(masks == v) for v in self.class_values]
        #masks = torch.stack(masks,axis=0).float()
        return img,masks
    
    def __len__(self):
        return len(self.ds)


train_Data=CityscapeData(train_dataset)
val_Data=CityscapeData(val_dataset)

BATCH_SIZE=12
train_loader = DataLoader(dataset=train_Data, batch_size=BATCH_SIZE,shuffle=True)
val_loader = DataLoader(dataset=val_Data, batch_size=BATCH_SIZE)


ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=35, 
    activation=ACTIVATION,
)

class F1(torch.nn.Module):
    __name__ = "f1"

    def __init__(self, threshold=None,mode="multiclass",num_classes=35,**kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.mode = mode
        self.class_values=[i for i in range(35)]
        self.classes=num_classes

    def forward(self, output, target):
        #masks = [(target == v) for v in self.class_values]
        #masks = torch.stack(masks,axis=1).int()
        output=output.argmax(axis=1)
        tp, fp, fn, tn = smp.metrics.get_stats(output, target, num_classes=self.classes,mode=self.mode, threshold=self.threshold)
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")
        return f1_score



loss = smp.losses.DiceLoss(mode="multiclass")
metrics = [F1(num_classes=35)]
optimizer = torch.optim.Adam([ dict(params=model.parameters(), lr=0.0001)])


train_epoch = train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)


if __name__=="__main__":
    max_score = 0
    for i in range(0, 200):  
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['f1']:
            max_score = valid_logs['f1']
            torch.save(model, './best_model.pth')
            print('Model saved!')
            
        if i == 100:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')