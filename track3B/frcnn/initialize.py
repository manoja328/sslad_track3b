import os
import torch
import torch.nn as nn
import torchvision
import ipdb
from tqdm import tqdm
import pickle
from torchvision.models.detection.transform import resize_boxes
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor , TwoMLPHead
from torchvision.models.detection.rpn import AnchorGenerator
import os.path as osp

import frcnn.transforms as T
from frcnn.frcnn_mod import ModifiedFasterRCNN
#%%

def get_fpn(num_classes):
    print("unmodified fpn backbone")
    # Setup model, optimizer and dummy criterion
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_res50_FRCNN(num_classes):
    print("res50 backbone")
    res50_model = torchvision.models.resnet50(pretrained=True)
    backbone = nn.Sequential(*list(res50_model.children())[:-2])
    backbone.out_channels = 2048
    model = ModifiedFasterRCNN(backbone, num_classes=num_classes)
    return model

def get_fpn_mod2(num_classes):
    print("fpn backbone distll")
    fpn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    backbone = fpn_model.backbone
    model = ModifiedFasterRCNN(backbone, num_classes=num_classes)
    return model


def get_fpn_mod(num_classes):
    print("fpn backbone minsize and max size changed")
    fpn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    backbone = fpn_model.backbone

    # anchor_sizes = ( (64,), (128,), (256,), (512,), (1024,))
    # aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    # rpn_anchor_generator = AnchorGenerator(
    #     anchor_sizes, aspect_ratios
    # )

    # model = ModifiedFasterRCNN(backbone, num_classes=num_classes,
    #                            rpn_anchor_generator=rpn_anchor_generator,
    #                            box_roi_pool=None
    #                            )

    model = ModifiedFasterRCNN(backbone, num_classes=num_classes,
                                min_size = 1280, max_size = 1920,
                               # rpn_anchor_generator=rpn_anchor_generator,
                               # box_roi_pool=None
                               )


    return model

    