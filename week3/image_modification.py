from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from itertools import product

coco = COCO('/home/alex/Desktop/university/M5VisualRecognition/M5-VisualRecognition/COCO/annotations/instances_val2017.json')
img_dir = '/home/alex/Desktop/university/M5VisualRecognition/M5-VisualRecognition/COCO/val2017'
destination_ids = [2923]
source_ids = [453302,97022]
annotation_ids = [2,0]

def _get_slice_bbox(arr):
    nonzero = np.nonzero(arr)
    return [(min(a), max(a)+1) for a in nonzero]

def crop_new(arr):
    slice_bbox = _get_slice_bbox(arr)
    return arr[[slice(*a) for a in slice_bbox]]

    return arr[[slice(*s) for s in slices]]

def extract_objects(source_ids , annotation_ids):
    mask_objs =[]
    bboxs = []
    for image_id, annotation_id in zip(source_ids, annotation_ids ):
        img = coco.imgs[image_id]
        image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))

        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        bbox = anns[annotation_id]['bbox']

        mask = coco.annToMask(anns[annotation_id])[:,:,np.newaxis]
        mask_obj = mask*image
        cropped = crop_new(mask_obj)
        plt.figure()
        plt.imshow(cropped)
        plt.show()
        mask_objs.append(cropped)
        bboxs.append(bbox)

    return mask_objs, bboxs

mask_objs, bboxs = extract_objects(source_ids, annotation_ids)

fold_dir = '/home/alex/Desktop/university/M5VisualRecognition/M5-VisualRecognition/week3/task_b'
def random_objects(destination_ids, mask_objs, fold_dir):
    count=0
    for image_id, mask_obj in product(destination_ids,mask_objs):

        img = coco.imgs[image_id]
        image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))

        plt.figure()
        plt.imshow(mask_obj)
        plt.show()

        obj_x, obj_y = mask_obj.shape[1], mask_obj.shape[0]
        max_x = image.shape[1]-obj_x
        max_y = image.shape[0]-obj_y

        x0 = np.random.randint(0,max_x)
        y0 = np.random.randint(0,max_y)
        mod_img = image.copy()
        mod_img[y0:y0+obj_y, x0:x0+obj_x, :] = mask_obj

        zeros = np.uint8(mod_img == 0)

        mod_img = np.maximum(mod_img,zeros*image)

        plt.figure()
        plt.imshow(mod_img)
        plt.show()

        plt.imsave(os.path.join(fold_dir,'{}_{}_{}.png'.format(image_id,count,2)),mod_img)
        count+=1

random_objects(destination_ids,mask_objs,fold_dir)
def co_ocurrence(destination_ids,mask_objs,bboxs,version, fold_dir):

    for image_id, mask_obj, bbox in zip(destination_ids,mask_objs, bboxs):
        img = coco.imgs[image_id]
        image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))

        plt.figure()
        plt.imshow(mask_obj)
        plt.show()
        print(bbox)
        obj_x, obj_y = mask_obj.shape[1], mask_obj.shape[0]
        mod_img = image.copy()

        disp = np.random.randint(0, 70)

        mod_img[int(bbox[1]) + disp :int(bbox[1]) + obj_y + disp, int(bbox[0]) + disp:int(bbox[0]) + obj_x + disp, :] = mask_obj

        zeros = np.uint8(mod_img == 0)

        mod_img = np.maximum(mod_img, zeros * image)

        plt.figure()
        plt.imshow(mod_img)
        plt.show()

        plt.imsave(os.path.join(fold_dir, '{}_{}.png'.format(image_id, version)), mod_img)

def modify_img(destination_ids, mask_objs, bboxs, version, fold_dir):

    for image_id, mask_obj, bbox in zip(destination_ids,mask_objs, bboxs):
        img = coco.imgs[image_id]
        image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
        mod_img_1 = np.random.randint(0,2,(image.shape[0],image.shape[1]),dtype='uint8')*255
        mod_img =np.zeros_like(image)
        mod_img[:,:,0] = mod_img_1
        mod_img[:, :, 1] = mod_img_1
        mod_img[:, :, 2] = mod_img_1

        print(bbox)
        obj_x, obj_y = mask_obj.shape[1], mask_obj.shape[0]

        mod_img[int(bbox[1]) :int(bbox[1]) + obj_y, int(bbox[0]) :int(bbox[0]) + obj_x, :] = mask_obj

        plt.figure()
        plt.imshow(mod_img)
        plt.show()

        plt.imsave(os.path.join(fold_dir, '{}_{}.png'.format(image_id, version)), mod_img)



