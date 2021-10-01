import math

import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from tqdm import tqdm
from augmentations import CIFAR10Policy
from .Network import network
from IncrementalDatasets import IncrementalCIFAR100
from torch.utils.data import DataLoader
from utils import ensure_dir, save_checkpoint, load_checkpoint
import copy
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_one_hot(target, num_class):
    one_hot = torch.zeros(target.shape[0], num_class).to(device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot


def adjust_learning_rate(optimizer, epoch, learning_rate, final_epoch, warmup=0):
    lr = learning_rate
    if warmup > 0 and epoch < warmup:
        lr = lr / (warmup - epoch)
    else:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup) / (final_epoch - warmup)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class iCaRLmodel:

    def __init__(self, numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate,
                 regularize=True):

        super(iCaRLmodel, self).__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = network(numclass, feature_extractor)
        self.exemplar_set = []
        self.class_mean_set = []
        self.numclass = numclass
        self.transform = transforms.Compose([  # transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.old_model = None

        self.train_transform = transforms.Compose([  # transforms.Resize(img_size),
            # Todo Make it changable by arguments
            CIFAR10Policy(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.24705882352941178),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        self.test_transform = transforms.Compose([  # transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        self.classify_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1.),
                                                      # transforms.Resize(img_size),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                           (0.2675, 0.2565, 0.2761))])

        self.train_dataset = IncrementalCIFAR100('dataset', transform=self.train_transform, download=True)
        self.test_dataset = IncrementalCIFAR100('dataset', test_transform=self.test_transform, train=False,
                                                download=True)

        self.batchsize = batch_size
        self.memory_size = memory_size
        self.task_size = task_size

        self.train_loader = None
        self.test_loader = None
        self.regularize = regularize

    # get incremental train data
    # incremental
    def beforeTrain(self):
        self.model.eval()
        classes = [self.numclass - self.task_size, self.numclass]
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        if self.numclass > self.task_size:
            self.model.Incremental_learning(self.numclass)
        self.model.train()
        self.model.to(device)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes, self.exemplar_set)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.batchsize)

        return train_loader, test_loader

    '''
    def _get_old_model_output(self, dataloader):
        x = {}
        for step, (indexs, imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                old_model_output = torch.sigmoid(self.old_model(imgs))
            for i in range(len(indexs)):
                x[indexs[i].item()] = old_model_output[i].cpu().numpy()
        return x
    '''

    # train model
    # compute loss
    # evaluate model
    def train(self, resume=False, task_id=1):
        if resume:
            print("loading from previous state dict")
            directory = './checkpoint'
            filename = directory + f'/task_id_{task_id}'
            self.model.load_state_dict(load_checkpoint(filename)['net'])
            return load_checkpoint(filename)['accuracy']
        accuracy = 0
        # opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)
        opt = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=3e-2)
        for epoch in range(1, self.epochs + 1):
            # if epoch == 48:
            #     if self.numclass == self.task_size:
            #         print(1)
            #         opt = optim.SGD(self.model.parameters(), lr=1.0 / 5, weight_decay=0.00001)
            #     else:
            #         for p in opt.param_groups:
            #             p['lr'] = self.learning_rate / 5
            #         # opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 5,weight_decay=0.00001,momentum=0.9,nesterov=True,)
            #     print("change learning rate:%.3f" % (self.learning_rate / 5))
            # elif epoch == 62:
            #     if self.numclass > self.task_size:
            #         for p in opt.param_groups:
            #             p['lr'] = self.learning_rate / 25
            #         # opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 25,weight_decay=0.00001,momentum=0.9,nesterov=True,)
            #     else:
            #         opt = optim.SGD(self.model.parameters(), lr=1.0 / 25, weight_decay=0.00001)
            #     print("change learning rate:%.3f" % (self.learning_rate / 25))
            # elif epoch == 80:
            #     if self.numclass == self.task_size:
            #         opt = optim.SGD(self.model.parameters(), lr=1.0 / 125, weight_decay=0.00001)
            #     else:
            #         for p in opt.param_groups:
            #             p['lr'] = self.learning_rate / 125
            #         # opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125,weight_decay=0.00001,momentum=0.9,nesterov=True,)
            #     print("change learning rate:%.3f" % (self.learning_rate / 100))
            adjust_learning_rate(opt, epoch, self.learning_rate, self.epochs, 5)
            total_loss = 0.
            total_images = 0
            for step, (indexs, images, target) in tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                                                       desc='Training'):
                images, target = images.to(device), target.to(device)
                # output = self.model(images)
                loss_value = self._compute_loss(indexs, images, target)
                opt.zero_grad()
                loss_value.backward()
                opt.step()
                total_loss += loss_value.item()
                total_images += images.size(0)
                # print('epoch:%d,step:%d,loss:%.3f' % (epoch, step, loss_value.item()))
            accuracy = self._test(self.test_loader, 1)
            avg_loss = total_loss / total_images if total_images != 0 else 1000
            wandb.log({
                "epoch": epoch,
                "training_avg_loss": avg_loss,
                "test_accuracy": accuracy
            })
            print('epoch:%d,accuracy:%.3f' % (epoch, accuracy))
        return accuracy

    def _test(self, testloader, mode):
        if mode == 0:
            print("compute NMS")
        self.model.eval()
        correct, total = 0, 0
        for setp, (indexs, imgs, labels) in tqdm(enumerate(testloader), desc="Testing", total=len(testloader)):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = self.model(imgs) if mode == 1 else self.classify(imgs)
            predicts = torch.max(outputs, dim=1)[1] if mode == 1 else outputs
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100 * correct / total
        self.model.train()
        return accuracy

    def _compute_loss(self, indexs, imgs, target):
        alpha = 0.1
        output = self.model(imgs)
        target = get_one_hot(target, self.numclass)
        output, target = output.to(device), target.to(device)
        loss = torch.zeros([1])
        if self.old_model == None:
            loss = F.binary_cross_entropy_with_logits(output, target)
        else:
            # old_target = torch.tensor(np.array([self.old_model_output[index.item()] for index in indexs]))
            old_target = torch.sigmoid(self.old_model(imgs))
            old_task_size = old_target.shape[1]
            target[..., :old_task_size] = old_target
            loss = F.binary_cross_entropy_with_logits(output, target)
        if self.regularize and self.old_model is not None:
           # print("Regularizing")
            for i in range(len(self.old_model.feature.classifier.blocks)):
                x = self.old_model.feature.classifier.blocks[i].self_attn.qkv.weight.data
                y = self.model.feature.classifier.blocks[i].self_attn.qkv.weight.data
                loss += alpha *  F.kl_div(y,x,reduction="batchmean")
        return loss

    # change the size of examplar
    def afterTrain(self, task_id, no_save=False):
        self.model.eval()
        m = int(self.memory_size / self.numclass)
        self._reduce_exemplar_sets(m)
        for i in range(self.numclass - self.task_size, self.numclass):
            print('construct class %s examplar:' % (i), end='')
            images = self.train_dataset.get_image_class(i)
            self._construct_exemplar_set(images, m)
        self.numclass += self.task_size
        self.compute_exemplar_class_mean()
        self.model.train()
        KNN_accuracy = self._test(self.test_loader, 0)
        print("NMS accuracy：" + str(KNN_accuracy.item()))
        # Saving old model
        self.old_model = copy.deepcopy(self.model)
        self.old_model.to(device)
        self.old_model.eval()
        if no_save:
            return
        directory = './checkpoint'
        filename = directory + f'/task_id_{task_id}'
        state = {
            'net': self.model.state_dict(),
            'task_id': task_id,
            'accuracy': KNN_accuracy,
            'optim': None,
        }
        ensure_dir(directory)
        save_checkpoint(state, filename)

    def _construct_exemplar_set(self, images, m):
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        exemplar = []
        now_class_mean = np.zeros((1, 512))

        for i in range(m):
            # shape：batch_size*512
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])

        print("the size of exemplar :%s" % (str(len(exemplar))))
        self.exemplar_set.append(exemplar)
        # self.exemplar_set.append(images)

    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
            print('Size of class %d examplar: %s' % (index, str(len(self.exemplar_set[index]))))

    def Image_transform(self, images, transform):
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data

    def compute_class_mean(self, images, transform):
        with torch.no_grad():
            x = self.Image_transform(images, transform).to(device)
            feature_extractor_output = F.normalize(self.model.feature_extractor(x).detach()).cpu().numpy()
        # feature_extractor_output = self.model.feature_extractor(x).detach().cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output

    def compute_exemplar_class_mean(self):
        self.class_mean_set = []
        for index in range(len(self.exemplar_set)):
            print("compute the class mean of %s" % (str(index)))
            exemplar = self.exemplar_set[index]
            # exemplar=self.train_dataset.get_image_class(index)
            class_mean, _ = self.compute_class_mean(exemplar, self.transform)
            class_mean_, _ = self.compute_class_mean(exemplar, self.classify_transform)
            class_mean = (class_mean / np.linalg.norm(class_mean) + class_mean_ / np.linalg.norm(class_mean_)) / 2
            self.class_mean_set.append(class_mean)

    def classify(self, test):
        result = []
        test = F.normalize(self.model.feature_extractor(test).detach()).cpu().numpy()
        # test = self.model.feature_extractor(test).detach().cpu().numpy()
        class_mean_set = np.array(self.class_mean_set)
        for target in test:
            x = target - class_mean_set
            x = np.linalg.norm(x, ord=2, axis=1)
            x = np.argmin(x)
            result.append(x)
        return torch.tensor(result)
