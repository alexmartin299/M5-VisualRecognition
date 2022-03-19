from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from dataloader import get_dataset_dicts_detection,get_dataset_dicts_segmentation


#first adapt the dataset and register it for each of the tasks
dataset_folder = '/home/alex/Desktop/university/M5VisualRecognition/M5-VisualRecognition/KITTI-MOTS'
for d in ['training','validation']:
    DatasetCatalog.register("kitti_mots_" + d + "_detection", lambda d=d: get_dataset_dicts_detection(dataset_folder, train_or_val=d))
    MetadataCatalog.get("kitti_mots_" + d + "_detection").set(thing_classes=["cars,pedestrians"])

for d in ['training','validation']:
    DatasetCatalog.register("kitti_mots_" + d + "_segmentation", lambda d=d: get_dataset_dicts_segmentation(dataset_folder, train_or_val=d))
    MetadataCatalog.get("kitti_mots_" + d + "_segmentation").set(thing_classes=["cars,pedestrians"])


#code for visualizing if works fine TERMINATE WINDOWS WITH 0
"""
kittimots_metadata_detection = MetadataCatalog.get("kitti_mots_train_detection")
dataset_dicts = get_dataset_dicts_detection(dataset_folder, train_or_val='training')
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=kittimots_metadata_detection, scale=1.2)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow('visualize',out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
"""

#have to use coco metrics for evaluation

#load coco weights of models
yaml_files = ["COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"]

for i in yaml_files:
    """hay que acabar de hacer el loop para evaluation"""
    pass


