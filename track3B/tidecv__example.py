# The bare-bones example from the README.
# Run coco_example.py first to get mask_rcnn_bbox.json
import matplotlib.pyplot as plt
from tqdm import tqdm
from tidecv import TIDE
from tidecv.data import Data
import tidecv.functions as f
import json
def read_json(file):
    with open(file) as f:
        return json.load(f)

dets = []
dets.extend(read_json("aa/result_val_0.json"))
dets.extend(read_json("aa/result_val_1.json"))
dets.extend(read_json("aa/result_val_2.json"))
dets.extend(read_json("aa/result_val_3.json"))


data = Data("FPN_largesize")
for det in dets:
    image = det['image_id']
    _cls = det['category_id']
    score = det['score']
    box = det['bbox'] if 'bbox' in det else None
    mask = det['segmentation'] if 'segmentation' in det else None
    data.add_detection(image, _cls, score, box, mask)

#%%

from detection_util import create_train_val_set
root = "../data/SSLAD-2D/labeled/"
train_sets, val_sets = create_train_val_set(root, avalanche=False)
categories = {
    1: "Pedestrain",
    2: "Cyclist",
    3: "Car",
    4: "Truck",
    5: "Tram (Bus)",
    6: "Tricycle"
}


#%%

gt = Data("gt", max_dets=100)
# image_lookup = {}
#
# for idx, image in enumerate(images):
#     image_lookup[image['id']] = image
#     data.add_image(image['id'], image['file_name'])


for cat in categories:
    gt.add_class(cat, categories[cat])

for ts in val_sets:
    L = len(ts)
    for i in tqdm(range(L)):
        target = ts._load_target(i)
        L = len(target['labels'])
        for idx in range(L):
            bbox = target['boxes'][idx].tolist()
            x,y, w, h = bbox[0], bbox[1] , bbox[2] - bbox[0], bbox[3] - bbox[1]
            image = target['image_id'].item()
            _class = target['labels'][idx].item()
            box = [x,y,w,h]
            mask = None
            gt.add_ground_truth(image, _class, box, mask)


#%%
tide = TIDE()
# tide.evaluate(gt, data, mode=TIDE.BOX)
tide.evaluate_range(gt, data, mode=TIDE.BOX)
tide.summarize()
tide.plot(out_dir=".")

#%%
run = tide.runs["FPN_largesize"]
keys = list(run.error_dict.keys())
print(keys)
class_errors = run.error_dict[keys[0]]
y_true = []
y_pred = []

for err in class_errors:
    t = err.gt['class']
    p = err.pred['class']
    y_true.append(t)
    y_pred.append(p)
#%%
categories = {
    1: "Pedestrain",
    2: "Cyclist",
    3: "Car",
    4: "Truck",
    5: "Tram (Bus)",
    6: "Tricycle"}

labels = list(categories.values())
labels = ['background'] + labels


from sklearn.metrics import confusion_matrix
c = confusion_matrix(y_true, y_pred,[1,2,3,4,5,6])
print (c)
