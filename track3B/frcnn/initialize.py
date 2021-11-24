import os
import torch
import torch.nn as nn
import torchvision
import ipdb
from tqdm import tqdm
import pickle
from torchvision.models.detection.transform import resize_boxes
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.rpn import AnchorGenerator
import os.path as osp

import frcnn.transforms as T
from frcnn.frcnn_mod import ModifiedFasterRCNN


# %%

def get_fpn(num_classes, **kwargs):
    print("unmodified fpn backbone")
    # Setup model, optimizer and dummy criterion
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, **kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class FastRCNNPredictor_temp(nn.Module):
    def __init__(self, in_channels, num_classes, temperature=2):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        self.temperature = temperature

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        if not self.training: #during test
            scores = torch.div(scores, self.temperature)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas

def get_fpn_temperature(num_classes, **kwargs):
    # Setup model, optimizer and dummy criterion
    temperature = 2
    print(f"temperature sclaed scores on fpn backbone temp: {temperature}")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, **kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor_temp(in_features, num_classes, temperature)
    return model

from torchvision.ops.poolers import MultiScaleRoIAlign
def get_fpn_scale(num_classes, **kwargs):
    print("unmodified with different scales backbone")
    # Setup model, optimizer and dummy criterion

    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3', '4'],
        output_size=7,
        sampling_ratio=2)

    kwargs['box_roi_pool'] = box_roi_pool
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, **kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_fpn_scale2(num_classes, **kwargs):
    print("unmodified 2 with different scales backbone")
    # Setup model, optimizer and dummy criterion

    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=['2', '3', '4'],
        output_size=7,
        sampling_ratio=2)

    kwargs['box_roi_pool'] = box_roi_pool
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, **kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_fpn_anchor(num_classes, **kwargs):
    print("anchor size changed fpn backbone")
    anchor_sizes = ((12,), (24,), (48,), (96,), (192,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )
    # Setup model, optimizer and dummy criterion
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                 rpn_anchor_generator=rpn_anchor_generator, **kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_fpn_thresh(num_classes, **kwargs):
    defaults = {
        "rpn_score_thresh": 0.05,
    }

    kwargs = {**defaults, **kwargs}
    print("Score thresh to 0.05 fpn backbone")
    # Setup model, optimizer and dummy criterion
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, **kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


from double_bbox_head import DoubleConvFCBBoxHead


def get_fpn_doublehead(num_classes, **kwargs):
    defaults = {
        "rpn_score_thresh": 0.00,
    }
    init_cfg = dict(
        type='Normal',
        override=[
            dict(type='Normal', name='fc_cls', std=0.01),
            dict(type='Normal', name='fc_reg', std=0.001),
            dict(
                type='Xavier',
                name='fc_branch',
                distribution='uniform')
        ])

    kwargs = {**defaults, **kwargs}
    print(f"fpn backbone Score thresh to {defaults['rpn_score_thresh']} + double head")
    # Setup model, optimizer and dummy criterion
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, **kwargs)
    out_channels = model.backbone.out_channels
    model.roi_heads.box_head = nn.Identity()  # dont need this since double head has this
    roi_feat_size = model.roi_heads.box_roi_pool.output_size

    doublehead = DoubleConvFCBBoxHead(num_convs=3,
                                      num_fcs=2,
                                      in_channels=out_channels,
                                      roi_feat_size=roi_feat_size,
                                      num_classes=num_classes,
                                      conv_cfg=None,
                                      conv_out_channels=512,
                                      fc_out_channels=1024,
                                      init_cfg=init_cfg,
                                      )

    print(doublehead)

    model.roi_heads.box_predictor = doublehead
    # if box_head is None:
    #     resolution = box_roi_pool.output_size[0]
    #     representation_size = 1024
    #     box_head = TwoMLPHead(
    #         out_channels * resolution ** 2,
    #         representation_size)
    # if box_predictor is None:
    #     representation_size = 1024
    #     box_predictor = FastRCNNPredictor(
    #         representation_size,
    #         num_classes)
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_fpn_frozen(num_classes):
    print("Frozen fpn backbone")
    # [layer0, layer1, layer2 , layer3, layer4]
    # if 5 all trainable
    # if 1 only the last trainable
    # Setup model, optimizer and dummy criterion
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=2)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    print("================================================")
    for idx, (pname, p) in enumerate(model.named_parameters()):
        if p.requires_grad:
            print(idx, pname, p.requires_grad)
    print("================================================")

    return model


from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.faster_rcnn import FasterRCNN


def get_res101_FRCNN(num_classes):
    print("res101 backbone trained on imagenet")
    pretrained = True
    trainable_backbone_layers = 5
    backbone = resnet_fpn_backbone('resnet101', pretrained, trainable_layers=trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes)
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
                               min_size=1280, max_size=1920,
                               # rpn_anchor_generator=rpn_anchor_generator,
                               # box_roi_pool=None
                               )

    return model
