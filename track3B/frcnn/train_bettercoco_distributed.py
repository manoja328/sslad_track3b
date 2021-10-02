import os
import torch
import torch.nn as nn
import torchvision
import ipdb
from tqdm import tqdm
import pickle
from torchvision.models.detection.transform import resize_boxes
from frcnn_mod import ModifiedFasterRCNN , FastRCNNPredictor
import frcnn.transforms as T
import os.path as osp
#%%

def get_transform(istrain=False):
     transforms = []
     transforms.append(T.ToTensor())
     if istrain:
         transforms.append(T.RandomHorizontalFlip(0.5))
     return T.Compose(transforms)


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
            image_id = int(targets[idx]['image_id'].item())
            orig_size = targets[idx]["size"]
            new_size = images.image_sizes[idx]
            box = self.edgeboxes[image_id]
            box = torch.Tensor(box).float()
            box = resize_boxes(box,orig_size,new_size)
            box = box.to(device)
            proposals.append(box)

        boxes = proposals
        losses = {}
        return boxes, losses



def get_model_FRCNN(num_classes):

    res50_model = torchvision.models.resnet50(pretrained=True)
    backbone = nn.Sequential(*list(res50_model.children())[:-2])
    backbone.out_channels = 2048
    
#    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
#    backbone.out_channels = 1280

    # FasterRCNN needs to know the number of output channels in a backbone.
    # For mobilenet_v2, it's 1280 so we need to add it here

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios 
    anchor_generator = None
    
    
    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)
    

    
    model = ModifiedFasterRCNN(backbone, num_classes,
                               rpn_anchor_generator=anchor_generator,
                               box_roi_pool=roi_pooler)
   
    model.rpn = FakeRegionProposalNetwork()
    
    return  model


def get_distillinfo(model,dl):   
    save = {}
    print ("dumping info ......")
    model.eval()
    with torch.no_grad():
        for ii, (images, targets) in tqdm(enumerate(dl),total=len(dl)):   
           images = list(image.to(device) for image in images)
           targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
           for image,target in zip(images,targets):
               image_id = '{0:06d}'.format(target['image_id'].item())
               info = model.get_data128([image], [target]) 
               save[image_id] = info  
        return save

def save_distillinfo(obj,file):    
    dirn = os.path.dirname(file)
    if not os.path.exists(dirn):
        os.mkdir(dirn)
    with open(file,"wb") as f:
        pickle.dump(obj,f)



    #
    #         model = get_model_FRCNN(num_classes)
    #
    #
    #         #get fc data from previous model
    #         fc_data =  model.roi_heads.box_predictor.state_dict()
    #         new_box_predictor = FastRCNNPredictor(1024,num_classes)
    #         for key in fc_data:
    #             ndim = fc_data[key].data.ndim
    #             s = fc_data[key].shape
    #             if ndim == 1:
    #                 new_box_predictor.state_dict()[key].data[:s[0]] = fc_data[key].detach()
    #             else:
    #                 new_box_predictor.state_dict()[key].data[:s[0],:s[1]] = fc_data[key].detach()
    #         new_box_predictor = new_box_predictor.to(device)
    #         model.roi_heads.box_predictor = new_box_predictor
    #         #try eval here too just ot verify
    #
    #
    # #%%
    #
    #         # if freezebn: model.apply(set_bn_eval)
    #         # warm_lr_scheduler = None
    #         # if epoch == 0:
    #         #     warmup_factor = 1. / 1000
    #         #     warmup_iters = min(1000, len(data_loader) - 1)
    #         #     warm_lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    #
    #         loss_epoch = {}
    #         header = 'Phase[{}] Epoch: [{}/{}]'.format(incriter,epoch,num_epochs)
    #         loss_name = ['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']
    #         for ii, (images, targets) in tqdm(enumerate(data_loader),total=len(data_loader),desc = header):
    #            optimizer.zero_grad()
    #            images = list(image.to(device) for image in images)
    #            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    #            # training
    #            loss_dict = model(images, targets)
    #            losses = sum(loss for loss in loss_dict.values())
    #            losses.backward()
    #            optimizer.step()
    #            if warm_lr_scheduler is not None:
    #                 warm_lr_scheduler.step()
    #            info = {}
    #            for name in loss_dict:
    #                info[name] = loss_dict[name].item()
    #
    #            writer.add_scalars("losses", info, epoch * iters_per_epoch + ii)
    #            if torch.isnan(losses):
    #                print ("NaN encountered!!! ", targets)
    #                ipdb.set_trace()
    #
    #

    