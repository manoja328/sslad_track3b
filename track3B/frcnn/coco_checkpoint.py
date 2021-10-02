import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import pickle
from pathlib import Path

import utils
from engine import evaluate
import transforms as T
from opt import parse_args


from frcnn_mod import ModifiedFasterRCNN, FastRCNNPredictor
from torchvision.models.detection.transform import resize_boxes
from train_bettercoco import dpr_to_normal , evaluate, getds , COCOLoader



class FakeRegionProposalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        print (" ----- Using fake region proposal boxes -----")
        with open("datasets/edboxes_coco2014trainval_2000.pkl","rb") as f:
            self.edgeboxes = pickle.load(f)

    def forward(self, images, features, targets=None):

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        num_images = len(images.tensors)
        device = images.tensors.device

        proposals = []
        for idx in range(num_images):
            image_id = '{0:06d}'.format(targets[idx]['image_id'].item())
            orig_size = targets[idx]["size"]
            new_size = images.image_sizes[idx]
            box = self.edgeboxes[image_id]
            box = torch.Tensor(box).float()
            box = resize_boxes(box, orig_size, new_size)
            box = box.to(device)
            proposals.append(box)

        boxes = proposals
        losses = {}
        return boxes, losses


def get_model_FRCNN(num_classes):
    # res50_model = torch.hub.load('facebookresearch/swav', 'resnet50')
    res50_model = torchvision.models.resnet50(pretrained=True)
    backbone = nn.Sequential(*list(res50_model.children())[:-2])
    backbone.out_channels = 2048

    anchor_generator = None

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)

    model = ModifiedFasterRCNN(backbone, num_classes,
                               rpn_anchor_generator=anchor_generator,
                               box_roi_pool=roi_pooler)

    model.rpn = FakeRegionProposalNetwork()

    return model


# %%
if __name__ == "__main__":
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = False

    args.ckpt_file = 'iter0_models_incr_coco/chkpt9.pth'
    classifier_ckpt = os.path.join(args.ckpt_file)
    core_model = get_model_FRCNN(num_classes=41)
    print(core_model)

    root, annFile = getds('val2014')
    dataset_test = COCOLoader(root, annFile, included=[])
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=4, shuffle=False,
        num_workers=2, collate_fn=utils.collate_fn)

    #evaluate will not works since only gt boxes are used there
    if os.path.exists(classifier_ckpt):
        core_model.eval()
        core_model = core_model.to(device)
        print("Reusing last checkpoint ", classifier_ckpt)
        load_tbs = utils.load_checkpoint(classifier_ckpt)
        core_model.load_state_dict(dpr_to_normal(load_tbs['state_dict']))
        # optimizer.load_state_dict(dpr_to_normal(load_tbs['optim_dict']))
        # eval the  checkpoint to verify
        evaluate(core_model, data_loader_test, device=device)
    else:
        print(classifier_ckpt, " not found!!")

