import json
import os

PATH_TO_DATASET = '../flickr30k_images'

with open(PATH_TO_DATASET + '/1.json', 'rb') as f:
    train_ann = json.load(f)
with open(PATH_TO_DATASET + '/test.json', 'rb') as f:
    test_ann = json.load(f)
with open(PATH_TO_DATASET + '/val.json', 'rb') as f:
    val_ann = json.load(f)


train_ids = [ann['filename'] for ann in train_ann]
test_ids = [ann['filename'] for ann in test_ann]
val_ids = [ann['filename'] for ann in val_ann]
dataset_ids = []
dataset_ids.extend(train_ids)
dataset_ids.extend(test_ids)
dataset_ids.extend(val_ids)
print(dataset_ids)

path_to_images = '../flickr30k_images/train/1'
for id in os.listdir(path_to_images):
    if id not in dataset_ids:
        os.remove(os.path.join(path_to_images,id))


