import os
import re
import argparse
import sys

import ipdb
import torchvision.models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from avalanche.benchmarks.scenarios.generic_benchmark_creation import create_multi_dataset_generic_benchmark
from avalanche.logging import TextLogger
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.training.plugins import ReplayPlugin

import torch
from detection_util import *
from detection_strategy import *

from avalanche.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.training.plugins import ReplayPlugin
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

from torchvision.models.detection.transform import GeneralizedRCNNTransform
# Flexible integration for any Python script
import wandb

def increase_minsize(model):
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    min_size = 1280
    max_size = 1920
    model.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
    return model


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='result',
                        help='Name of the result files')
    parser.add_argument('--root', default="../data",
                        help='Root folder where data is stored.')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Num workers to use for dataloading.')
    parser.add_argument('--test', action='store_true',
                        help='If set model will be evaluated on test set, else on validation set')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--store', action='store_true',
                        help='If set the predicition files required for submission will be created')
    parser.add_argument('--store_model', action='store_true',
                        help="Stores model if specified. Has no effect is store is not set")
    parser.add_argument('--load_model', type=str, default=None,
                        help='Loads model with given name. Model should be stored in current folder')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--fake', action='store_true', help="simulate on small data")
    args = parser.parse_args()

    ######################################
    #                                    #
    #  Editing below this line allowed   #
    #                                    #
    ######################################

    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    data_root = args.root = f"{args.root}/SSLAD-2D/labeled"

    epochs = 10
    batch_size = 8
    num_classes = 7

    # 1. Start a W&B run
    wandb.init(project='iccv', entity='ma7583')

    from frcnn.initialize import get_fpn_mod2, get_fpn, get_fpn_mod
    model = get_fpn(num_classes)

    model = increase_minsize(model)

    #so add another plugin after that
    # optimizer.step()
    # if warm_lr_scheduler is not None:
    #     warm_lr_scheduler.step()

    # if args.test_only: #should we do this the first paper suggested to do this
    #     model = different_test_mode(model)

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()

    # eval_plugin = EvaluationPlugin(
    #     accuracy_metrics(
    #         minibatch=True, epoch=True, experience=True, stream=True),
    #     loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    #     forgetting_metrics(experience=True),
    #     loggers=[interactive_logger])

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.00005, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
    lr_scheduler_plugin = LRSchedulerPlugin(lr_scheduler)
    if args.fake:
        args.mem_size = 10
    else:
        args.mem_size = 83  # 83 x 3 = 249


    # warmup_factor = 1. / 1000
    # # warmup_iters = min(1000, len(data_loader) - 1)
    # warmup_iters = 50 #chose here what is good
    # warm_lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    # warm_scheduler_plugin = LRSchedulerPlugin(warm_lr_scheduler)

    # replay_plugin = ReplayPlugin(mem_size)
    # plugins = [DetectionStrategyPlugin(), lr_scheduler_plugin, replay_plugin]
    plugins = [DetectionStrategyPlugin(args.mem_size, wandb), lr_scheduler_plugin]
    # plugins = [DetectionStrategyPlugin(args.mem_size), lr_scheduler_plugin, warm_scheduler_plugin]
    print('Arguments')
    printargs = vars(args)
    for arg in printargs.keys():
        line = '%20s : %20s' % (arg, printargs[arg])
        print(line)

    ######################################
    #                                    #
    # No editing below this line allowed #
    #                                    #
    ######################################

    if args.load_model is not None:
        model.load_state_dict(torch.load(f"./{args.load_model}"))

    if args.fake:
        train_datasets, val_datasets = create_train_val_set_small(data_root, validation_proportion=0.1)
    else:
        # Setup Benchmark
        train_datasets, val_datasets = create_train_val_set(data_root, validation_proportion=0.1)

    if args.test:
        eval_datasets, _ = create_test_set_from_json(data_root)
    else:
        eval_datasets = val_datasets

    benchmark = create_multi_dataset_generic_benchmark(train_datasets=train_datasets, test_datasets=eval_datasets)

    # Setup evaluation and logging
    test_split = "test" if args.test else "val"

    result_file = open(f"./{args.name}_{test_split}.txt", "w")
    logger = TextLogger(result_file)
    gt_path = f"{args.root}/annotations/instance_{test_split}.json"
    # change this a little here
    if not args.store:
        store = None
    else:
        os.makedirs(args.name, exist_ok=True)
        store = os.path.join(args.name, f"result_{test_split}")
    eval_plugin = EvaluationPlugin(detection_metrics(gt_path, experience=True, store=store, pred_only=args.test),
                                   loggers=logger)

    # Create strategy.
    criterion = empty
    strategy = DetectionBaseStrategy(
        model, optimizer, criterion, train_mb_size=batch_size, train_epochs=epochs,
        eval_mb_size=batch_size, device=device, evaluator=eval_plugin, plugins=plugins)

    if args.test_only:
        results = strategy.eval(benchmark.test_stream, num_workers=args.num_workers)
        task_mean_map = sum(float(re.findall(r'\d+.\d+', rv)[-1]) for rv in results.values()) / len(results)
        result_file.writelines([f"Task mean MAP: {task_mean_map:.3f} \n"])

    else:
        for train_exp in benchmark.train_stream:
            strategy.train(train_exp, num_workers=args.num_workers)

        # #add different model here
        # model = different_test_mode(model)

        # Only evaluate at the end of training
        results = strategy.eval(benchmark.test_stream, num_workers=args.num_workers)
        task_mean_map = sum(float(re.findall(r'\d+.\d+', rv)[-1]) for rv in results.values()) / len(results)
        result_file.writelines([f"Task mean MAP: {task_mean_map:.3f} \n"])

    if args.store_model:
        torch.save(model.state_dict(), f'./{args.name}.pt')

    result_file.close()


if __name__ == '__main__':
    main()
