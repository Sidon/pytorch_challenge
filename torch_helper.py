# Main imports
import torch
from torchvision import datasets, models
from torchvision import transforms as tv_transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class MyNet:
    def __init__(self, trained_name, fc_layers = [512, 256], dropout=0.2, out_features = 102):
        # Additional variables will be used to track training progress for easier re-loads, plotting, etc.

        self.fc_layers = fc_layers
        self.dropout = dropout
        self.out_features = out_features
        self.__trained_name = trained_name

        self.__tracking = {}
        self.__tracking['last_epoch'] = 0
        self.__tracking['min_loss'] =  99
        self.__tracking['learn_rates'] = []
        self.__tracking['validation_loss'] = []
        self.__tracking['training_loss'] = []
        self.__tracking['accuracy_rate'] = []
        self.__tracking['trained_name'] = trained_name
        self.__tracking['los_name'] = None
        self.__tracking['optmizer_name'] = None

        self.__trained_models = {
            'ResNet152': {'size': 2048, 'model': models.resnet152},
            'Inception': {'size': 2048, 'model': models.inception_v3},
            'DenseNet161': {'size': 2208, 'model': models.densenet161}
        }

        # Import trained model
        self.__trained_model = self.__trained_models[self.__trained_name]['model'](pretrained=True)

        # Create the full connected layer
        self._fc = self.__fc(self.__trained_models[trained_name]['size'], fc_layers, self.dropout, self.out_features)

        # Replace fc in trained model
        self.__trained_model.fc = self.fc

    def __fc(self, in_features, hidden_features = [512, 256], dropout=0.2, out_features = 102 ):
        fc = nn.Sequential()  # type: object
        n = 1
        for h in hidden_features:
            fc_name = 'fc' + str(n)
            relu_name = 'relu' + str(n)
            fc.add_module(fc_name, nn.Linear(in_features, h))
            fc.add_module(relu_name, nn.ReLU())
            in_features = h
            n+=1
        fc.add_module('out', nn.Linear(in_features, out_features))
        return fc

    @property
    def fc(self):
        return self._fc

    @property
    def model(self):
        return self.__trained_model

    @property
    def model(self):
        return self.__import_trained_model() if self.__trained_model is None else self.__trained_model

    def __import_trained_model(self):
        return  self.__trained_models[self.__trained_name]['model'](pretrained=True)



class Transforms:
    normalize = ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    crop = 224
    @classmethod
    def validation(cls, resize=256, crop=224, normalize = normalize):
        compose = tv_transforms.Compose([tv_transforms.Resize(resize), tv_transforms.CenterCrop(crop),
                                         tv_transforms.ToTensor(),
                                         tv_transforms.Normalize(normalize[0],normalize[1])])
        return compose

    @classmethod
    def train(cls, augment=[tv_transforms.RandomRotation(30), tv_transforms.RandomHorizontalFlip(),
                            tv_transforms.CenterCrop(224), tv_transforms.RandomVerticalFlip()], normalize=normalize):

        lst_compose = augment + [tv_transforms.ToTensor(), tv_transforms.Normalize( normalize[0], normalize[1]) ]
        return tv_transforms.Compose(lst_compose)


class Dataset:
    @staticmethod
    def dataset(data_dir, transform):
        return datasets.ImageFolder(data_dir, transform = transform)


class Loaders:
    num_workers=0
    batch_size=20
    @classmethod
    def loader(cls, dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers):
        return torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size = batch_size,
                                           num_workers = num_workers)


