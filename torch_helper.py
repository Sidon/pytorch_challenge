# Main imports
import torch
from torchvision import datasets, models
from torchvision import transforms as tv_transforms
from torch.utils.data.sampler import SubsetRandomSampler
# import matplotlib.pyplot as plt
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim


class MyNet:
    def __init__(self, trained_name, fc_layers = None, out_features = 102, optimizer=None):
        # Additional variables will be used to track training progress for easier re-loads, plotting, etc.


        # self.__tracking = {}
        # self.__tracking['last_epoch'] = 0
        # self.__tracking['min_loss'] =  99
        # self.__tracking['learn_rates'] = []
        # self.__tracking['validation_loss'] = []
        # self.__tracking['training_loss'] = []
        # self.__tracking['accuracy_rate'] = []
        # self.__tracking['trained_name'] = trained_name
        # self.__tracking['los_name'] = None
        # self.__tracking['optmizer_name'] = None

        # for saving model
        self.id_model = trained_name

        self.__trained_models = {
            'vgg11':    {'model': models.vgg11, 'change_model': self.__vgg},
            'vgg11_bn': {'model': models.vgg11_bn, 'change_model': self.__vgg},
            'vgg13':    {'model': models.vgg13, 'change_model': self.__vgg},
            'vgg13_bn': {'model': models.vgg13_bn, 'change_model': self.__vgg},
            'vgg16': {'model': models.vgg16, 'change_model': self.__vgg},
            'vgg16_bn': {'model': models.vgg16_bn, 'change_model': self.__vgg},
            'vgg19': {'model': models.vgg19, 'change_model': self.__vgg},
            'vgg19_bn': {'model': models.vgg19_bn, 'change_model': self.__vgg},
            'resnet101': {'model': models.resnet101, 'change_model': self.__resnet},
            'resnet152': {'model': models.resnet152, 'change_model': self.__resnet},
            'resnet18': {'model': models.resnet18, 'change_model': self.__resnet},
            'resnet34': {'model': models.resnet34, 'change_model': self.__resnet},
            'resnet50': {'model': models.resnet50, 'change_model': self.__resnet},
            'densenet121': {'model': models.densenet121, 'change_model': self.__densenet},
            'densenet161': {'model': models.densenet161, 'change_model': self.__densenet},
            'densenet169': {'model': models.densenet169, 'change_model': self.__densenet},
            'densenet201': {'model': models.densenet201, 'change_model': self.__densenet},
            'inception': {'model': models.inception_v3, 'change_model': self.__inception},
        }

        # Criar o optimizer em cad rede
        self.optimizer = None if optimizer is None else optimizer

        # Import trained model
        __model = self.__trained_models[trained_name]['model'](pretrained=True)

        # Change last layer of trained model
        self.__trained_model = self.__trained_models[trained_name]['change_model'](__model, fc_layers, out_features)

        # Config parameters of the trained model
        self.__config_trained_model()


    def __vgg(self, model, fc=None, out_features=102):
        if not fc is None:
            model.classifier = fc
        else:
            model.classifier[6].out_features = out_features
        return model

    def __densenet(self, model, fc=None, out_features=102):
        if not fc is None:
            model.classifier = clf
        else:
            model.classifier.out_features = out_features
        return model

    def __inception(self, model, fc=None, out_features=102):
        if not fc is None:
            model.fc = fc
        else:
            model.fc.out_features = out_features
        return model

    def __squeeznet(self, model, fc=None, out_features=102):
        if not fc is None:
            model.fc = fc
        else:
            model.fc.out_features = out_features
        return model

    def __resnet(self, model, fc=None, out_features=102):
        if not fc is None:
            model.fc = fc
        else:
            model.fc.out_features = out_features
        return model

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

    def __config_trained_model(self):
        for param in self.__trained_model.parameters():
             param.requires_grad = False

    def create_optmizer(self, optimizer=None, **kwargs):
        optimizer = self.optimizer if optimizer==None else optimizer
        return optimizer(**kwargs)



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




