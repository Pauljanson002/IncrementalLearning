import argparse
import os
import pickle
import random
import time

import numpy as np
import torch
import wandb
from torch import nn

from Learner import Learner
from models import model_pool
from models.util import create_model

from IncrementalDataset import IncrementalDataset


def parse_option():
    parser = argparse.ArgumentParser('Training arguments');
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='test_batch_size',help='Size of test batch)')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--workers', type=int, default=0, help='num of workers to use')

    # optimizer
    parser.add_argument('--optimizer', type=str, default="adamw", help='optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--tags', type=str, default="gen0, ssl", help='add tags for the experiment')

    # dataset
    parser.add_argument('--sess', type=int, default=0, help='start session id')
    parser.add_argument('--model', type=str, default='convnet4', choices=model_pool)
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('--num_class', type=int, default=100, help='number of classes')
    parser.add_argument('--num_task', type=int, default=10, help='total number of tasks')

    # savefolders
    parser.add_argument('--model_path', type=str, default='save/', help='path to save model')
    parser.add_argument('--test_case', type=str, default='test1', help='test case')
    parser.add_argument('--data_root', type=str, default='../../Datasets/CIFAR10', help='path to data root')
    parser.add_argument('--savepoint',type=str , default='./saves',help='Saves')

    # hyper parameters
    parser.add_argument('--gamma', type=float, default=2, help='loss cofficient for ssl loss')
    parser.add_argument('--res', type=int, default=64, help='resolution of input image')

    opt = parser.parse_args()
    opt.class_per_task = opt.num_class // opt.num_task
    opt.random_classes = False
    opt.validation = 0
    opt.memory = 2000
    opt.mu = 1
    opt.beta = 1.0
    opt.r = 1
    opt.use_cuda = True
    opt.checkpoint = opt.savepoint
    return opt


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args = parse_option()

    # wandb
    wandb.init(project='Incremental learning test')
    wandb.config.update(args)
    wandb.run.save()

    inc_dataset = IncrementalDataset(
        dataset_name=args.dataset,
        args=args,
        random_order=False,
        shuffle=True,
        seed=1,
        batch_size=args.batch_size,
        workers=args.workers,
        validation_split=0,
        increment=10,
    )
    model = create_model(args.model, args.num_class, args.dataset)
    wandb.watch(model)
    print(f'Total params: {sum(p.numel() for p in model.parameters())} ')

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        print("Training on cuda")
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("No cuda")
        device = torch.device('cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    start_sess = args.sess
    memory = None

    for sess_id in range(start_sess, args.num_task):
        args.sess = sess_id

        if (sess_id == 0):
            torch.save(model.state_dict(), os.path.join(args.savepoint, 'base_model.pth.tar'))
            mask = {}

        if (start_sess == sess_id and start_sess != 0):
            inc_dataset._current_task = sess_id
            with open(args.savepoint + "/sample_per_task_testing_" + str(args.sess - 1) + ".pickle", 'rb') as handle:
                sample_per_task_testing = pickle.load(handle)
            inc_dataset.sample_per_task_testing = sample_per_task_testing
            args.sample_per_task_testing = sample_per_task_testing

        if sess_id > 0:
            path_model = os.path.join(args.savepoint, 'session_' + str(sess_id - 1) + '_model_best.pth.tar')
            prev_best = torch.load(path_model)
            model.load_state_dict(prev_best)

            with open(args.savepoint + "/memory_" + str(args.sess - 1) + ".pickle", 'rb') as handle:
                memory = pickle.load(handle)

        task_info, train_loader, val_loader, test_loader, for_memory = inc_dataset.new_task(memory)
        print(task_info)
        print(inc_dataset.sample_per_task_testing)
        args.sample_per_task_testing = inc_dataset.sample_per_task_testing
        main_learner = Learner(model=model, args=args, trainloader=train_loader, testloader=test_loader,
                               use_cuda=args.use_cuda)

        main_learner.learn()
        memory = inc_dataset.get_memory(memory, for_memory)

        acc_task = main_learner.meta_test(main_learner.best_model, memory, inc_dataset)

        with open(args.savepoint + "/memory_" + str(args.sess) + ".pickle", 'wb') as handle:
            pickle.dump(memory, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(args.savepoint + "/acc_task_" + str(args.sess) + ".pickle", 'wb') as handle:
            pickle.dump(acc_task, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(args.savepoint + "/sample_per_task_testing_" + str(args.sess) + ".pickle", 'wb') as handle:
            pickle.dump(args.sample_per_task_testing, handle, protocol=pickle.HIGHEST_PROTOCOL)

        time.sleep(10)


if __name__ == '__main__':
    main()
