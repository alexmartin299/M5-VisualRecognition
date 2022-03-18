# Some basic setup:
# Setup detectron2 logger
import detectron2
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

imgs_sequences = ['0001','0003','0004','0000','0005','0009','0011']
yaml_files = ["COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"]

for i in yaml_files:
    cfg = get_cfg()
    #inference with faster R-CNN
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(i))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(i)
    predictor = DefaultPredictor(cfg)
    for img in imgs_sequences:
        im = cv2.imread("/home/alex/Desktop/university/M5VisualRecognition/M5-VisualRecognition/KITTI-MOTS/training/{}/000000.png".format(img))
        outputs = predictor(im)
        # We can use `Visualizer` to draw the predictions on the image.
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow('detected_image',out.get_image()[:, :, ::-1])
        cv2.waitKey(0)


