import ipdb
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from tqdm import tqdm

"""
A strategy pulgin can change the strategy parameters before and after 
every important fucntion. The structure is shown belown. For instance, self.before_train_forward() and
self.after_train_forward() will be called before and after the main forward loop of the strategy.

The training loop is organized as follows::
        **Training loop**
            train
                train_exp  # for each experience
                    adapt_train_dataset
                    train_dataset_adaptation
                    make_train_dataloader
                    train_epoch  # for each epoch
                        train_iteration # for each minibatch
                            forward
                            backward
                            model update

        **Evaluation loop**
        The evaluation loop is organized as follows::
            eval
                eval_exp  # for each experience
                    adapt_eval_dataset
                    eval_dataset_adaptation
                    make_eval_dataloader
                    eval_epoch  # for each epoch
                        eval_iteration # for each minibatch
                            forward
                            backward
                            model update
"""

#since you can make your own model so you can change update function there
#the limit is only 250 images ( or objects from those)
#so RODEO could easily store those ,,,, but what about bounding boxes 
#LOOK at icarl ....
#or other model examples 

import copy
import torch
import torch.nn as nn
import itertools
import random

class CutPaste(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image1, target1, image2, target2):
        if random.random() < self.prob:
            boxes = target1['boxes'].tolist()
            labels = target1['labels'].tolist()
            for idx in range(len(target2['boxes'])): #add paste objects and labels
                bbox = target2['boxes'][idx]
                l = target2["labels"][idx].item()
                if l in {1, 2, 6}:
                    image1[bbox[1:3, 0:2]] = image2[bbox[1:3, 0:2]] #paste from target2 x1,y1,x2,y2
                    boxes.append(bbox.tolist())
                    labels.append(l)
        target1['boxes'] = torch.Tensor(boxes)
        target1['labels'] = torch.Tensor(labels)
        return image1, target1


from avalanche.benchmarks.utils import AvalancheConcatDataset, \
    AvalancheTensorDataset, AvalancheSubset
from collections import Counter


def get_data128(model, images, targets=None):
    model.eval()
    with torch.no_grad():
        images, targets = model.transform(images, targets)
        features = model.backbone(images.tensors)
        proposals, proposal_losses = model.rpn(images, features, targets)
        len_proposals = [len(p) for p in proposals]
        # idxes = 128
        # proposals = [p[:idxes] for p in old_proposals]
        box_features = model.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
        box_features = model.roi_heads.box_head(box_features)
        class_logits, box_regression = model.roi_heads.box_predictor(box_features)
        return len_proposals, class_logits, box_regression



def distill_bbLoss(bbox, target):
    return torch.mean((bbox - target) ** 2)


def distill_CELoss(logits, target):
    # subtract mean over class dimension from un-normalized logits
    logits = logits - torch.mean(logits, dim=0, keepdim=True)
    target = target - torch.mean(target, dim=0, keepdim=True)
    class_distillation_loss = torch.mean((logits - target) ** 2)
    return class_distillation_loss

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_lr(optimizer , lr):
    for g in optimizer.param_groups:
        g['lr'] = lr

class DetectionStrategyPlugin(StrategyPlugin):

    def __init__(self, memory_size, wandb, alpha=1.0, temperature=2.0):
        super(DetectionStrategyPlugin).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.prev_model = None
        self.wandb = wandb
        self.memory_size = memory_size
        self.x_memory = []
        self.y_memory = []

    def clear_memory(self):
        self.x_memory = []
        self.y_memory = []


    def _distillation_loss(self, out, prev_out):
        """
        Compute distillation loss between output of the current model and
        and output of the previous (saved) model.
        """

        log_p = torch.log_softmax(out / self.temperature, dim=1)
        q = torch.softmax(prev_out / self.temperature, dim=1)
        res = torch.nn.functional.kl_div(log_p, q, reduction='batchmean')
        return res

    def _iccv_distillation_loss(self, old, new):
        old_logits, old_boxes = old
        new_logits, new_boxes = new
        celoss = distill_CELoss(new_logits, old_logits)
        bbloss = distill_bbLoss(new_boxes, old_boxes)
        return celoss + bbloss

    def penalty(self, strategy, images, targets, alpha):
        """
        Compute weighted distillation loss.
        """

        if self.prev_model is None:
            return 0
        else:
            print ("------------- starting distillaiton .................")
            current_model = strategy.model
            # ipdb.set_trace()
            new_len_proposals, new_logits, new_boxes = get_data128(current_model, images, targets)
            # old_logits = new_logits.clone() + 0.1
            # old_boxes = new_boxes.clone() + 0.002
            with torch.no_grad():
                old_len_proposals, old_logits, old_boxes = get_data128(self.prev_model, images, targets)

            # new_len_proposals
            # [2000, 2000, 2000, 2000, 2000, 2000]
            # new_logits.shape
            # torch.Size([12000, 7])
            # ipdb> new_boxes.shape
            # torch.Size([12000, 28])
            #-----------
            #assume proposals are sorted in descending order of proposal scores
            #i.e more probable objets are first
            soft = torch.softmax(new_logits, dim=1)
            sample_indices = []
            prev = 0
            #process for each image
            for next_img_idx in new_len_proposals:
                # sort over bg score default is ascending
                values, sorted_idx = torch.sort(soft[prev:prev + next_img_idx,0], dim=0)
                N = 128
                topN = sorted_idx[:N]  #select top 128 with the least bg proabilities
                nsample = 64  # sample 64 out of 128
                idx = torch.randperm(N)[:nsample] #sample nsample out of N --> indexes
                sample_indices.append(topN[idx].tolist()) #get the sample
                prev = prev + next_img_idx

            #exclude bg + new_class and include only  previous class logits/boxes
            new_class_logits = new_logits[sample_indices, 1:]
            old_class_logits = old_logits[sample_indices, 1:]

            new_class_boxes = new_boxes[sample_indices, 4:]
            old_class_boxes = old_boxes[sample_indices, 4:]

            old = (old_class_logits, old_class_boxes)
            new = (new_class_logits,  new_class_boxes)

            dist_loss = self._iccv_distillation_loss(old, new)
            # dist_loss = self._distillation_loss(new_class_logits, old_class_logits)
            print ("------------- ending distillaiton .................")
            return alpha * dist_loss

    # def penalty(self, strategy, images, targets, alpha):
    #     """
    #     Compute weighted distillation loss.
    #     """
    #
    #     if self.prev_model is None:
    #         return 0
    #     else:
    #         print ("------------- starting distillaiton .................")
    #         current_model = strategy.model
    #         # ipdb.set_trace()
    #         new_len_proposals, new_logits, new_boxes = current_model.get_data128(images, targets)
    #         # old_logits = new_logits.clone() + 0.1
    #         # old_boxes = new_boxes.clone() + 0.002
    #         with torch.no_grad():
    #             old_len_proposals, old_logits, old_boxes = self.prev_model.get_data128(images, targets)
    #
    #         # new_len_proposals
    #         # [2000, 2000, 2000, 2000, 2000, 2000]
    #         # new_logits.shape
    #         # torch.Size([12000, 7])
    #         # ipdb> new_boxes.shape
    #         # torch.Size([12000, 28])
    #         #-----------
    #         #assume proposals are sorted in descending order of proposal scores
    #         #i.e more probable objets are first
    #         soft = torch.softmax(new_logits, dim=1)
    #         sample_indices = []
    #         prev = 0
    #         #process for each image
    #         for next_img_idx in new_len_proposals:
    #             # sort over bg score default is ascending
    #             values, sorted_idx = torch.sort(soft[prev:prev + next_img_idx,0], dim=0)
    #             N = 128
    #             topN = sorted_idx[:N]  #select top 128 with the least bg proabilities
    #             nsample = 64  # sample 64 out of 128
    #             idx = torch.randperm(N)[:nsample] #sample nsample out of N --> indexes
    #             sample_indices.append(topN[idx].tolist()) #get the sample
    #             prev = prev + next_img_idx
    #
    #         #exclude bg + new_class and include only  previous class logits/boxes
    #         new_class_logits = new_logits[sample_indices, 1:]
    #         old_class_logits = old_logits[sample_indices, 1:]
    #
    #         new_class_boxes = new_boxes[sample_indices, 4:]
    #         old_class_boxes = old_boxes[sample_indices, 4:]
    #
    #         old = (old_class_logits, old_class_boxes)
    #         new = (new_class_logits,  new_class_boxes)
    #
    #         dist_loss = self._iccv_distillation_loss(old, new)
    #         # dist_loss = self._distillation_loss(new_class_logits, old_class_logits)
    #         print ("------------- ending distillaiton .................")
    #         return alpha * dist_loss

    def before_training(self, strategy: 'BaseStrategy', **kwargs):
        # print("reached here before training ....")
        pass
        # before training .. get pQ features from model using the before traing  

    def before_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        # print("task label", strategy.experience.task_labels, "experience dataset", len(strategy.experience.dataset))
        #import ipdb; ipdb.set_trace()
        pass

    def before_train_dataset_adaptation(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_train_dataset_adaptation(self, strategy: 'BaseStrategy', **kwargs):
        # start from the second experience
        if strategy.training_exp_counter > 0:
            old_task_labels = [strategy.training_exp_counter - 1] * len(self.x_memory)
            memory = AvalancheTensorDataset(
                torch.stack(self.x_memory).cpu(),
                self.y_memory,
                # self.y_memory, task_labels=old_task_labels,
                transform=None, target_transform=None)

            # strategy.adapted_dataset = AvalancheConcatDataset((strategy.adapted_dataset, memory))
            strategy.adapted_dataset = AvalancheConcatDataset((strategy.experience.dataset, memory))
            a,b,c = strategy.training_exp_counter, strategy.epoch, len(strategy.adapted_dataset)
            print(f"Adapted dataset Experience:{a} Epoch:{b} Len:{c}")



    def before_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_forward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_forward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_backward(self, strategy: 'BaseStrategy', **kwargs):
        """
        Add distillation loss
        """
        # other things there:
        #         'loss',
        #         'mb_it',
        #         'mb_output',
        #         'mb_task_id',
        #         'mb_x',
        #         'mb_y',
        #         'mbatch',

        # ipdb > strategy.loss
        # tensor(2.8923, device='cuda:0', grad_fn= < AddBackward0 >)
        # ipdb > strategy.loss.item()
        # 2.8922817707061768
        # ipdb > strategy.loss
        # tensor(2.8923, device='cuda:0', grad_fn= < AddBackward0 >)
        # ipdb > strategy.mb_output
        # {'loss_classifier': tensor(2.0625, device='cuda:0', grad_fn= < NllLossBackward >), 'loss_box_reg': tensor(
        #     0.7555, device='cuda:0', grad_fn= < DivBackward0 >), 'loss_objectness': tensor(0.0371, device='cuda:0',
        #     grad_fn= < BinaryCrossEntropyWithLogitsBackward >), 'loss_rpn_box_reg': tensor(
        #     0.0372, device='cuda:0', grad_fn= < DivBackward0 >)}

        # ipdb.set_trace()

        # alpha = self.alpha[strategy.training_exp_counter] \
        #     if isinstance(self.alpha, (list, tuple)) else self.alpha

        # strategy.training_exp_counter
        # strategy.epoch

        # #only for the first epoch of each experience
        # if strategy.epoch == 0:
        #     # should ideally distill on replay samples or whole previous dataloader which is expensive
        #     #replay images with most / best boxes
        #     imgs, targets, task_labels = strategy.mbatch
        #     penalty = self.penalty(strategy, imgs, targets, alpha)
        #     strategy.loss += penalty

        # distill on replay samples
        imgs, targets, task_labels = strategy.mbatch
        penalty = self.penalty(strategy, imgs, targets, self.alpha)
        strategy.loss += penalty


        tid = strategy.training_exp_counter
        ep = strategy.epoch
        batch_loss = strategy.loss.item()
        lr = get_lr(strategy.optimizer)
        self.wandb.log({f"loss-{tid}": batch_loss, "loss": batch_loss, "epoch": ep, "experience": tid, "lr": lr})



    def after_backward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        pass
        #we can do replay
        #replace and then again do forward pass using x + replay_x
        #replay samples is only 250
                
    def before_update(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_update(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        """
        Save a copy of the model after each experience
        """
        self.prev_model = copy.deepcopy(strategy.model)

        self.construct_exemplar_set(strategy)

    # def construct_exemplar_set(self, strategy):
    #     tid = strategy.training_exp_counter
    #
    #     # self.clear_memory() # always clear previous memory since it is added to adapted datasets
    #
    #     dataset = strategy.experience.dataset # will it have adapted dataset or clean?
    #     #if not clean then only need to select the new
    #     L = len(dataset)
    #     stats = {}
    #     position = {} #pos in dataset
    #     print("Finding images with class 1,2,6")
    #     for i in tqdm(range(L), desc=f"Exp {tid}"):
    #         _, target, _ = dataset[i]
    #         image_id = target['image_id'].item()
    #         labels = target['labels'].tolist()
    #         n = {6: 0}
    #         for k in labels:
    #             if k in n:
    #                 n[k] += 1
    #         stats[image_id] = sum(n.values())
    #         position[image_id] = i
    #     sorted_stats = dict(sorted(stats.items(), key=lambda item: -item[1])) #sort descending
    #
    #
    #     print(f"experience: {tid} len: {L}")
    #     nadded = 0
    #     for image_id in sorted_stats:
    #         if nadded == self.memory_size: #do not let exceed memroy size / 4 ( no. of tasks)
    #             break
    #         imgs, targets, task_labels = dataset[position[image_id]]
    #         ex_image_id = targets['image_id'].item()
    #         assert ex_image_id == image_id, "wrong order..."
    #         if nadded < self.memory_size: #do not let exceed memroy size / 4 ( no. of tasks)
    #             self.x_memory.append(imgs)
    #             self.y_memory.append(targets)
    #             nadded +=1


    def construct_exemplar_set(self, strategy):
        tid = strategy.training_exp_counter

        # self.clear_memory() # always clear previous memory since it is added to adapted datasets

        dataset = strategy.experience.dataset # will it have adapted dataset or clean?
        #if not clean then only need to select the new
        L = len(dataset)
        stats = {}
        position = {} #pos in dataset
        print("Finding images with class 1,2,6")
        for i in tqdm(range(L), desc=f"Exp {tid}"):
            _, target, _ = dataset[i]
            image_id = target['image_id'].item()
            labels = target['labels'].tolist()
            n = {1: 0, 2: 0, 6: 0}
            for k in labels:
                if k in n:
                    n[k] += 1
            stats[image_id] = sum(n.values())
            position[image_id] = i
        sorted_stats = dict(sorted(stats.items(), key=lambda item: -item[1])) #sort descending


        print(f"experience: {tid} len: {L}")
        nadded = 0
        for image_id in sorted_stats:
            if nadded == self.memory_size: #do not let exceed memroy size / 4 ( no. of tasks)
                break
            imgs, targets, task_labels = dataset[position[image_id]]
            ex_image_id = targets['image_id'].item()
            assert ex_image_id == image_id, "wrong order..."
            if nadded < self.memory_size: #do not let exceed memroy size / 4 ( no. of tasks)
                self.x_memory.append(imgs)
                self.y_memory.append(targets)
                nadded +=1

        #
        # print(f"experience: {tid} len: {L}")
        # s = []
        # for idx, batch in enumerate(dataset):
        #     if idx == self.memory_size: #do not let exceed memroy size / 4 ( no. of tasks)
        #         break
        #     imgs, targets, task_labels = batch
        #     image_id = target['image_id'].item()
        #     if image_id in sorted_stats:
        #         if idx < self.memory_size: #do not let exceed memroy size / 4 ( no. of tasks)
        #             self.x_memory.append(imgs)
        #             self.y_memory.append(targets)
        #     # s.extend(targets['labels'].tolist())
        # c = Counter(s)
        # print("stats ", {k: c[k] for k in sorted(c)})
        # print("-----------------------")


    def after_training(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_eval(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_eval_dataset_adaptation(self, strategy: 'BaseStrategy',
                                       **kwargs):
        pass

    def after_eval_dataset_adaptation(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_eval_exp(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_eval_exp(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_eval(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_eval_iteration(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_eval_forward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_eval_forward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_eval_iteration(self, strategy: 'BaseStrategy', **kwargs):
        pass