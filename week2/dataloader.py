import os
import numpy as np
from detectron2.structures import BoxMode
import PIL.Image as Image
import cv2

def get_files_txt(dataset_folder, train_or_val = 'training'):
    """ Returns filenames where there are the folder of the sequences
    with the images and the text instances"""
    images_path = os.path.join(dataset_folder,train_or_val)
    instances_path = os.path.join(dataset_folder,'instances_txt_'+train_or_val)
    training_image_paths = []
    training_instances_path = []

    for folder in os.listdir(images_path):
        training_image_paths.append(os.path.join(images_path,folder))
        training_instances_path.append(os.path.join(instances_path,str(folder)+'.txt'))

    training_image_folders = sorted(training_image_paths)
    training_instances_txts = sorted(training_instances_path)

    return [(folder,txt) for folder,txt in zip(training_image_folders, training_instances_txts)]

def get_files_instances(dataset_folder, train_or_val = 'training'):
    """ Returns filenames where there are the folder of the sequences
    with the images and the text instances"""
    images_path = os.path.join(dataset_folder,train_or_val)
    instances_path = os.path.join(dataset_folder,'instances_'+train_or_val)
    training_image_paths = []
    training_instances_path = []

    for folder in os.listdir(images_path):
        training_image_paths.append(os.path.join(images_path,folder))
        training_instances_path.append(os.path.join(instances_path,folder))

    training_image_folders = sorted(training_image_paths)
    training_instances_folders = sorted(training_instances_path)

    return [(folder,txt) for folder,txt in zip(training_image_folders, training_instances_folders)]

def read_annotations_detection(gt,map_classes):
    patterns = list(np.unique(gt))[1:-1]

    objs = []
    for pattern in patterns:
        coords = np.argwhere(gt==pattern)

        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0)

        bbox = [y0, x0, y1, x1]

        obj = {
            "bbox": bbox,
            "bbox_mode":BoxMode.XYXY_ABS,
            "category_id": map_classes[int(np.floor(gt[coords[0][0]][coords[0][1]]/1e3))],
            "iscrowd": 0
        }

        objs.append(obj)

    return objs

def read_annotations_segmentation(gt,map_classes):
    patterns = list(np.unique(gt))[1:-1]

    objs = []
    for pattern in patterns:
        coords = np.argwhere(gt==pattern)

        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0)

        bbox = [y0, x0, y1, x1]

        copy = gt.copy()
        copy[gt==pattern] = 255
        copy[gt!=pattern] = 0
        copy = np.asarray(copy,np.uint8)

        contours, _ = cv2.findContours(copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        contour = [np.reshape(contour,(contour.shape[0],2)) for contour in contours]
        contour = np.asarray([item for tree in contour for item in tree])
        px = contour[:,0]
        py = contour[:,1]
        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        poly = [p for x in poly for p in x]

        if len(poly) < 6:
            continue


        obj = {
            "bbox": bbox,
            "bbox_mode":BoxMode.XYXY_ABS,
            "segmentation": [poly],
            "category_id": map_classes[int(np.floor(gt[coords[0][0]][coords[0][1]]/1e3))],
            "iscrowd": 0
        }

        objs.append(obj)

    return objs

def load_dataset(dataset_folder,map_classes,train_or_val,type='detection'):

    dataset_dicts = []
    for folder_img, folder_instance in get_files_instances(dataset_folder, train_or_val):

        for img_name, instance_name in zip(os.listdir(folder_img),os.listdir(folder_instance)):

            record = {}
            filename = os.path.join(folder_img, img_name)
            gt_filename = os.path.join(folder_instance,instance_name)

            gt = np.asarray(Image.open(gt_filename))

            height, width = gt.shape[:]

            record["file_name"] = filename
            record["image_id"] = filename
            record["height"] = height
            record["width"] = width
            if type  =='detection':
                record["annotations"] = read_annotations_detection(gt,map_classes)
            elif type == 'segmentation':
                record["annotations"] = read_annotations_segmentation(gt, map_classes)
            dataset_dicts.append(record)

    return dataset_dicts
