from detectron2.utils.logger import setup_logger
setup_logger()

import cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from dataloader import load_dataset




#setup cfg to perform inference and evaluation
def setup_cfg(path_to_model):
    cfg = get_cfg()
    #inference with faster R-CNN
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(path_to_model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(path_to_model)
    results_dir = './output/qualitative'
    cfg.OUTPUT_DIR = results_dir + '/'
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


#code for visualizing if the data conversion works fine TERMINATE WINDOWS WITH 0
"""dataset_dicts = load_dataset(dataset_folder, map_classes, train_or_val='validation')
for d in random.sample(dataset_dicts, 5):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.2)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow('visualize',out.get_image()[:, :, ::-1])
    cv2.waitKey(0)"""

#Perform quantitative evaluation with COCO evaluator
def coco_evaluator(cfg, predictor,register_name):

    evaluator = COCOEvaluator(register_name, output_dir="./output")
    val_loader = build_detection_test_loader(cfg, register_name)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

# Plots inferences of the model in random samples of the dataset
def qualitative_results(yaml_files, dataset_functions,metadata):
    for i,dataset_function in zip(yaml_files,dataset_functions):
        cfg, predictor = setup_cfg(i)
        dataset_dict = dataset_function(dataset_folder, train_or_val='validation')
        plot_examples(dataset_dict,predictor,metadata)

map_classes = {1:2,2:0}
#first adapt the validation dataset and register it for each of the tasks
dataset_folder = '/home/alex/Desktop/university/M5VisualRecognition/M5-VisualRecognition/KITTI-MOTS'

#load coco weights of models
#yaml_files = [ "COCO-InstanceSegmentation/faster_rcnn_R_50_FPN_1x.yaml","COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"]
yaml_files = [ "COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml","COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml"]
#yaml_files = [ "COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml","COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"]
types = [ 'detection', 'segmentation']

for (round , i), type in zip(enumerate(yaml_files),types):
    cfg, predictor = setup_cfg(i)

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    things_classes = metadata.thing_classes

    for d in ['validation']:
        DatasetCatalog.register("kitti_mots_" + d +"_"+type,
                                lambda d=d: load_dataset(dataset_folder, map_classes, train_or_val=d,type=type))
        MetadataCatalog.get("kitti_mots_" + d + "_"+type).set(thing_classes=things_classes)

        metadata = MetadataCatalog.get('kitti_mots_validation_' +type)

    cfg.DATASETS.TRAIN = ('kitti_mots_validation_'+type,)
    cfg.DATASETS.TEST = ('kitti_mots_validation_'+type,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(things_classes)

    evaluator = COCOEvaluator("kitti_mots_validation_"+type, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "kitti_mots_validation_"+type)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))




