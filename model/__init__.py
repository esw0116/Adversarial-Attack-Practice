import torch
from torch import nn
import torchvision
from importlib import import_module


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.name = args.model
        self.built_in_model = False

        if self.name in ['resnet50', 'densenet121', 'inception_v3']:
            self.built_in_model = True
            if self.name == 'resnet50':
                self.model = torchvision.models.resnet50(pretrained=True).to(self.device)
            elif self.name == 'densenent121':
                self.model = torchvision.models.densenet121(pretrained=True).to(self.device)
            elif self.name == 'inception_v3':
                self.model = torchvision.models.inception_v3(pretrained=True).to(self.device)

        else:
            module = import_module('model.' + args.model.lower())
            self.model = module.make_model(args).to(self.device)

        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

    def forward(self, *input):
        return self.model(*input)
