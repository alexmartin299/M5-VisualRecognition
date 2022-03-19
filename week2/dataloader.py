import os
import numpy as np
import pycocotools.mask as rletools
import cv2
from detectron2.structures import BoxMode
import glob
import PIL.Image as Image

class SegmentedObject:
  def __init__(self, mask, class_id, track_id):
    self.mask = mask
    self.class_id = class_id
    self.track_id = track_id


def filename_to_frame_nr(filename):
    assert len(filename) == 10, "Expect filenames to have format 000000.png, 000001.png, ..."
    return int(filename.split('.')[0])


def load_image(filename, id_divisor=1000):
    img = np.array(Image.open(filename))
    obj_ids = np.unique(img)

    objects = []
    mask = np.zeros(img.shape, dtype=np.uint8, order="F")  # Fortran order needed for pycocos RLE tools
    for idx, obj_id in enumerate(obj_ids):
        if obj_id == 0:  # background
            continue
        mask.fill(0)
        pixels_of_elem = np.where(img == obj_id)
        mask[pixels_of_elem] = 1
        objects.append(SegmentedObject(
            rletools.encode(mask),
            obj_id // id_divisor,
            obj_id
        ))

    return objects

def load_images_for_folder(path):
  files = sorted(glob.glob(os.path.join(path, "*.png")))

  objects_per_frame = {}
  for file in files:
    objects = load_image(file)
    frame = filename_to_frame_nr(os.path.basename(file))
    objects_per_frame[frame] = objects

  return objects_per_frame

def load_txt(path):

  objects_per_frame = {}
  track_ids_per_frame = {}  # To check that no frame contains two objects with same id
  combined_mask_per_frame = {}  # To check that no frame contains overlapping masks
  with open(path, "r") as f:
    for line in f:
      line = line.strip()
      fields = line.split(" ")

      frame = int(fields[0])
      if frame not in objects_per_frame:
        objects_per_frame[frame] = []
      if frame not in track_ids_per_frame:
        track_ids_per_frame[frame] = set()
      if int(fields[1]) in track_ids_per_frame[frame]:
        assert False, "Multiple objects with track id " + fields[1] + " in frame " + fields[0]
      else:
        track_ids_per_frame[frame].add(int(fields[1]))

      class_id = int(fields[2])
      if not(class_id == 1 or class_id == 2 or class_id == 10):
        assert False, "Unknown object class " + fields[2]

      mask = {'size': [int(fields[3]), int(fields[4])], 'counts': fields[5].encode(encoding='UTF-8')}
      if frame not in combined_mask_per_frame:
        combined_mask_per_frame[frame] = mask
      elif rletools.area(rletools.merge([combined_mask_per_frame[frame], mask], intersect=True)) > 0.0:
        assert False, "Objects with overlapping masks in frame " + fields[0]
      else:
        combined_mask_per_frame[frame] = rletools.merge([combined_mask_per_frame[frame], mask], intersect=False)
      objects_per_frame[frame].append(SegmentedObject(
        mask,
        class_id,
        int(fields[1])
      ))

  return objects_per_frame

def get_files(dataset_folder, train_or_val = 'training'):
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


def get_dataset_dicts_detection(dataset_folder, train_or_val):
    """ Returns dict to register for the case of detection
      dataset_folder: folder where KITTI-MOTS is with specific structure
      train_or_val: string with value 'training' or 'validation'"""
    dataset_dicts = []
    for folder, txt in get_files(dataset_folder, train_or_val):
        # get data folder and its corresponding txt file
        # load the annotations for the folder
        annotations = load_txt(txt)
        image_paths = sorted(os.listdir(folder))
        for (image_path, (file_id, objects)) in zip(image_paths, list(annotations.items())):
            record = {}

            filename = os.path.join(folder, image_path)
            height,width = cv2.imread(filename).shape[:2]

            record["file_name"] = filename
            record["image_id"] = filename
            record["height"] = height
            record["width"] = width

            objs = []
            for obj in objects:
                if obj.track_id != 10000:
                    category_id = obj.class_id
                    bbox = rletools.toBbox(obj.mask)

                    obj_dic = {
                        "bbox" : list(bbox),
                        "bbox_mode" : BoxMode.XYWH_ABS,
                        "category_id" : category_id
                    }
                    objs.append(obj_dic)

            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts

def get_dataset_dicts_segmentation(dataset_folder, train_or_val):
    """ Returns dict to register for the case of segmentation
    dataset_folder: folder where KITTI-MOTS is with specific structure
    train_or_val: string with value 'training' or 'validation'"""
    dataset_dicts = []
    for folder, txt in get_files(dataset_folder, train_or_val):
        # get data folder and its corresponding txt file
        # load the annotations for the folder
        annotations = load_txt(txt)
        image_paths = sorted(os.listdir(folder))
        for (image_path, (file_id, objects)) in zip(image_paths, list(annotations.items())):
            record = {}

            filename = os.path.join(folder, image_path)
            height,width = cv2.imread(filename).shape[:2]

            record["file_name"] = filename
            record["image_id"] = filename
            record["height"] = height
            record["width"] = width

            objs = []
            for obj in objects:
                if obj.track_id != 10000:
                    category_id = obj.class_id
                    bbox = rletools.toBbox(obj.mask)
                    mask = obj.mask

                    obj_dic = {
                        "bbox" : list(bbox),
                        "bbox_mode" : BoxMode.XYWH_ABS,
                        "category_id" : category_id,
                        "segmentation" : mask
                    }
                    objs.append(obj_dic)

            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts


