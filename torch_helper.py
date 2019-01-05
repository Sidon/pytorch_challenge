# Main imports
import torch
from torchvision import datasets, models
from torchvision import transforms as tv_transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from torch import nn
from collections import OrderedDict


class MyNet:
    def __init__(self, trained_name, fc_layers = None, out_features = 102, optimizer=optim.SGD):

        # to saving model
        self.trained_model = trained_name

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

        self.__clf = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 1024)),
            ('relu1', nn.ReLU()),
            ('drop1', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(1024, 256)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(256, 102)),
            ('output', nn.LogSoftmax(dim=1))]))

        # Criar o optimizer em cada rede
        self.optimizer = None if optimizer is None else optimizer

        # Import trained model
        __model = self.__trained_models[trained_name]['model'](pretrained=True)

        # Freeze the trainde model
        self.freeze_model(__model)

        # Change last layer of trained model
        self.__trained_model = self.__trained_models[trained_name]['change_model'](__model, fc_layers, out_features)

        # Config parameters of the trained model
        # self.__config_trained_model()


    def __vgg(self, model, fc=None, out_features=102):
        if not fc is None:
            model.classifier = fc
        else:
            model.classifier[6].out_features = out_features
        return model

    def __densenet(self, model, fc=None, out_features=102):
        if not fc is None:
            model.classifier = fc
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

    # @property
    # def model(self):
    #     return self.__import_trained_model() if self.__trained_model is None else self.__trained_model

    def __import_trained_model(self):
        return  self.__trained_models[self.__trained_name]['model'](pretrained=True)

    def __config_trained_model(self):
        if 'fc' in dir(self.__trained_model):
            self.__trained_model.fc = self.__clf
        if 'classifier' in dir(self.trained_model):
            self.__trained_model.classifier = self.__clf

    def freeze_model(self, model):
        for param in model.parameters():
             param.requires_grad = False

    def create_optmizer(self, optimizer=None, parameters=None, **kwargs):
        optimizer = self.optimizer if optimizer==None else optimizer
        if not kwargs:
            kwargs = {'lr':0.001}
        if parameters is None:
            if 'fc' in dir(self.__trained_model):
                parameters = self.__trained_model.fc.parameters()
            elif 'classifier' in dir(self.__trained_model):
                parameters = self.__trained_model.classifier.parameters()
            else:
                parameters = self.__trained_model.parameters()

        return optimizer(parameters, **kwargs)


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




