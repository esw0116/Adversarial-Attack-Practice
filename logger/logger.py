import torch
import imageio
import numpy as np
import os
import datetime
from scipy import misc
import skimage.color as sc

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class Logger:
    def __init__(self, args):
        self.args = args
        self.acc_log = torch.Tensor()
        self.loss_log = torch.Tensor()

        if args.load == '.':
            if args.save == '.':
                args.save = datetime.datetime.now().strftime('%Y%m%d_%H:%M')
            self.dir = 'experiment/' + args.save
        else:
            self.dir = 'experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.acc_log = torch.load(self.dir + '/acc_log.pt')
                self.loss_log = torch.load(self.dir + '/loss_log.pt')
                print('Continue from epoch {}...'.format(len(self.acc_log)))

        if args.reset:
            os.system('rm -rf {}'.format(self.dir))
            args.load = '.'

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            if not os.path.exists(self.dir + '/model'):
                os.makedirs(self.dir + '/model')
        if not os.path.exists(self.dir + '/result/'+self.args.data_test):
            print("Creating dir for saving images...", self.dir + '/result/'+self.args.data_test)
            os.makedirs(self.dir + '/result/'+self.args.data_test)

        print('Save Path : {}'.format(self.dir))

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write('From epoch {}...'.format(len(self.acc_log)) + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def write_log(self, log):
        print(log)
        self.log_file.write(log + '\n')

    def save(self, trainer, epoch, is_best):
        trainer.model.save(self.dir, is_best)
        torch.save(self.loss_log, os.path.join(self.dir, 'loss_log.pt'))
        torch.save(self.acc_log, os.path.join(self.dir, 'acc_log.pt'))
        torch.save(trainer.optimizer.state_dict(), os.path.join(self.dir, 'optimizer.pt'))
        self.plot_loss_log(epoch)
        self.plot_acc_log(epoch)

    def save_images(self, filename, save_list, scale):
        filename = '{}/result/{}/{}_x{}_'.format(self.dir, self.args.data_test, filename, scale)
        if self.args.task == 'D':
            postfix = ['GT', 'Pred']
        else:
            postfix = ['SR', 'Blended']
        for img, post in zip(save_list, postfix):
            img = img[0].data
            img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype('float32')
            if self.args.task =='D':
                img = img/3.
            if img.shape[2] == 1:
                img = img.squeeze(axis=2)
            img = np.round(255*img).astype('uint8')
            imageio.imwrite('{}{}.png'.format(filename, post), img)

    def start_log(self, train=True):
        if train:
            self.loss_log = torch.cat((self.loss_log, torch.zeros(1)))
        else:
            self.acc_log = torch.cat((self.acc_log, torch.zeros(1)))

    def report_log(self, item, train=True):
        if train:
            self.loss_log[-1] += item
        else:
            self.acc_log[-1] += item

    def end_log(self, n_div, train=True):
        if train:
            self.loss_log[-1].div_(n_div)
        else:
            self.acc_log[-1].div_(n_div)

    def plot_loss_log(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title('Loss Graph')
        plt.plot(axis, self.loss_log.numpy())
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(self.dir, 'loss.pdf'))
        plt.close(fig)

    def plot_acc_log(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title('Accuracy Graph')
        plt.plot(axis, self.acc_log.numpy())
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy(%)')
        plt.grid(True)
        plt.savefig(os.path.join(self.dir, 'accuracy.pdf'))
        plt.close(fig)

    def done(self):
        self.log_file.close()
