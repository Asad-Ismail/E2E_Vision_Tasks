# Transformers for Vision and NLP tasks

# Vision Transformers

## DETR
## Train Instance segemntation network on custom dataset
Originally DETR was trained on OD and Panoptic segmentaiton task. We train instance segmentation task on DETR and evaluate how it performs
The steps involved in preparing the custom dataset are 
1) Custom data set is converted to COCO format using convert_to_coco.ipynb. We can change it as per new new format of dataset.
2) Second the COCO format instance segmentation network is converted to Panoptic segmentation format using panopticapi/converters/detection2panoptic_coco_format.py
3) Train and Test the network

