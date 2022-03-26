from detectron2.utils.logger import setup_logger
setup_logger()

import cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from dataloader import load_dataset
import os
import torch


def setup_cfg(path_to_model):
    cfg = get_cfg()
    #inference with faster R-CNN
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(path_to_model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(path_to_model)
    results_dir = './output'
    cfg.OUTPUT_DIR = results_dir
    predictor = DefaultPredictor(cfg)

    return cfg, predictor

def plot_examples(dataset_dicts,predictor,metadata):

    for d in random.sample(dataset_dicts, 10):
        im = cv2.imread(d["file_name"])
        outputs = predictor(
            im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=1.2
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow('visualization',out.get_image()[:, :, ::-1])
        cv2.waitKey(0)

def coco_evaluator(cfg, predictor, register_name):

    evaluator = COCOEvaluator(register_name, output_dir="./output")
    val_loader = build_detection_test_loader(cfg, register_name)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

# Plots inferences of the model in random samples of the dataset
def qualitative_results(yaml_files, dataset_functions, metadata):
    for i, dataset_function in zip(yaml_files, dataset_functions):
        cfg, predictor = setup_cfg(i)
        dataset_dict = dataset_function(dataset_folder, train_or_val='validation')
        plot_examples(dataset_dict, predictor, metadata)

def evaluate():
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("kitti_mots_validation_" + type, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "kitti_mots_validation_" + type)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))


map_classes = {1:0, 2:1}
things_classes = ['Car', 'Pedestrian']
# first adapt the validation dataset and register it for each of the tasks
dataset_folder = '/home/alex/Desktop/university/M5VisualRecognition/M5-VisualRecognition/KITTI-MOTS'
#load coco weights of model
yaml_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
#Specify the type of work it is going to be performe detection or segmentation
type = 'segmentation'



DatasetCatalog.register("kitti_mots_training_" + type,
                        lambda d='training': load_dataset(dataset_folder, map_classes, train_or_val=d, type=type))
MetadataCatalog.get("kitti_mots_training_" + type).set(thing_classes=things_classes)
metadata_train = MetadataCatalog.get("kitti_mots_training_" + type)
DatasetCatalog.register("kitti_mots_validation_" + type,
                        lambda d='validation': load_dataset(dataset_folder, map_classes, train_or_val=d, type=type))
MetadataCatalog.get("kitti_mots_validation_" + type).set(thing_classes=things_classes)
metadata_val = MetadataCatalog.get("kitti_mots_validation_" + type)


#Setup config of trainer
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(yaml_file))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml_file)

results_dir = '/home/alex/Desktop/university/M5VisualRecognition/M5-VisualRecognition/week2/output'
cfg.OUTPUT_DIR = results_dir
cfg.DATASETS.TRAIN = ("kitti_mots_training_" + type,)
cfg.DATASETS.TEST = ("kitti_mots_validation_" + type,)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(things_classes)
cfg.SOLVER.IMS_PER_BATCH = 1

cfg.SOLVER.BASE_LR = 0.00001
cfg.SOLVER.MAX_ITER = 2000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.DEVICE = "cuda"
cfg.INPUT.MASK_FORMAT = 'bitmask'

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model

predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("kitti_mots_validation_" + type, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "kitti_mots_validation_" + type)
print(inference_on_dataset(predictor.model, val_loader, evaluator))







