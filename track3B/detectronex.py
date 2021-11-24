# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

#%%
# Create config
cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"

cfg.MODEL.DEVICE = "cuda:2"

# Create predictor
predictor = DefaultPredictor(cfg)
#
# # get image
# im = cv2.imread("./input.jpg")
#
# # Make prediction
# outputs = predictor(im)

#%%


from detection_util import *
from detection_strategy import *

data_root =  f"../data/SSLAD-2D/labeled"
# Setup Benchmark
train_datasets, val_datasets = create_train_val_set(data_root, validation_proportion=0.1)

#%%

from detectron2.data import DatasetCatalog
from detectron2.structures import BoxMode

dataset = val_datasets[0]._dataset
id2annot = dataset.img_annotations

def my_dataset_function():
    L = len(dataset)
    new_list = []
    for i in range(L):
        entry = dataset._load_target(i)
        new_entry = {}
        new_entry['image_id'] = entry['image_id'].item()
        image_details = id2annot[new_entry['image_id']]
        new_entry['file_name'] = os.path.join(data_root , "train" , image_details['file_name'])
        new_entry['height'] = image_details['height']
        new_entry['width'] = image_details['width']
        new_entry['annotations'] = []
        for idx in range(len(entry['boxes'])):
            box = {}
            box['bbox'] = entry['boxes'][idx].tolist()
            box['bbox_mode'] = BoxMode.XYXY_ABS
            box['iscrowd'] = entry['iscrowd'][idx].item()
            box['category_id'] = entry['labels'][idx].item()
            new_entry['annotations'].append(box)

        new_list.append(new_entry)
    return new_list

from detectron2.data import DatasetCatalog
name = "hi6_dataset"
DatasetCatalog.register(name, my_dataset_function)
#%%

# later, to access the data:
dataset_dicts = DatasetCatalog.get(name)
print (name, len(dataset_dicts))

#%%
import random
for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)






