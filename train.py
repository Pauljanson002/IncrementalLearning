import argparse
import os

import wandb

import models
from IncrementalTrainingApproach.iCaRL import iCaRLmodel
from models.ResNet import resnet18_cbam
import numpy as np
import torch
torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

numclass = 10
feature_extractor = resnet18_cbam()
img_size = 32
batch_size =32
task_size = 10
memory_size = 2000
learning_rate = 2.0

config = dict(
    img_size=32,
    batch_size=128,
    task_size=10,
    memory_size=2000,
    learning_rate=2.0,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parameters for the script")
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--online',
        action='store_true',
    )
    parser.add_argument(
        '--project_name',
        type=str,
        default='project_null'
    ),
    parser.add_argument(
        '--feature_extractor',
        type=str,
        default='resnet'
    )
    os.environ['WANDB_MODE'] = 'offline'
    args = parser.parse_args()
    if args.online:
        os.environ['WANDB_MODE'] = 'online'
    feature_extractor = models.get_feature_extractor(args.feature_extractor)
    model = iCaRLmodel(numclass, feature_extractor, batch_size, task_size, memory_size, args.epochs, learning_rate)
    # model.model.load_state_dict(torch.load('model/ownTry_accuracy:84.000_KNN_accuracy:84.000_increment:10_net.pkl'))
    config["epochs"] = args.epochs
    # wandb.watch(model.model)

    for i in range(10):
        config["task_id"] = i+1
        # print(config)
        run = wandb.init(
            project=args.project_name,
            reinit=True,
            config=config,
        )
        model.beforeTrain()
        accuracy = model.train()
        model.afterTrain(accuracy,task_id=i+1)
        run.finish()
