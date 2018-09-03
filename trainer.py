import os
import math
import time
import imageio
import decimal

import numpy as np
from scipy import misc

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from tensorboardX import SummaryWriter

import utils

class Trainer:
    def __init__(self, args, loader, my_model):
        self.args = args
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.optimizer = self.make_optimizer()
        self.scheduler = self.make_scheduler()
        self.writer = SummaryWriter()
        self.loss = nn.CrossEntropyLoss()

        if args.load != '.' and not args.test_only:
            self.optimizer.load_state_dict(torch.load(os.path.join(ckp.dir, 'optimizer.pt')))
            for _ in range(len(ckp.loss_log)):
                self.scheduler.step()

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        return optim.Adam(self.model.parameters(), **kwargs)

    def make_scheduler(self):
        kwargs = {'step_size': self.args.lr_decay, 'gamma': self.args.gamma}
        return lrs.StepLR(self.optimizer, **kwargs)

    def train(self):
        self.scheduler.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log('Epoch {:3d} with Lr {:.2e}'.format(epoch, decimal.Decimal(lr)))

        self.model.train()
        self.ckp.start_log()
        for batch, (image, target, _) in enumerate(self.loader_train):
            image = image.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(image)
            loss = self.loss(image, target)

            self.ckp.report_log(loss.item())
            loss.backward()
            self.optimizer.step()

            if (batch+1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\tLoss : {:.5f}'.format(
                    (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                    self.ckp.loss_log[-1] / (batch + 1)))

        self.ckp.end_log(len(self.loader_train))

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        self.ckp.start_log(train=False)
        with torch.no_grad():
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (image, target, filename) in enumerate(tqdm_test):
                filename = filename[0]
                image = image.to(self.device)
                target = target.to(self.device)
                name = self.model.get_model().name
                pred = self.model(image)
                pred = torch.argmax(pred, dim=1)
                correct = (pred == target)
                acc = correct.mean()
                acc = torch.stack(acc)
                self.ckp.report_log(acc, train=False)
                save_list = [cm, prediction]
                if self.args.save_images:
                    self.ckp.save_images(filename, save_list, self.args.scale)

            self.ckp.end_log(len(self.loader_test), train=False)
            best = self.ckp.acc_log[:, 8].max(0)
            self.ckp.write_log('[{}]\taverage accuracy: {:.3f} % (Best: {:.3f} % @epoch {})\n'.format(
                                self.args.data_test, self.ckp.acc_log[-1], best[0], best[1] + 1))
            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
