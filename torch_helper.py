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
    def __init__(self, trained_name, fc_layers = None, out_features = 102):
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

        self.__trained_models = {
            'vgg11': {'model': models.vgg11, 'change_model': self.__vgg},
            'vgg11_bn': {'model': models.vgg11_bn, 'change_model': self.__vgg},
            'vgg13': {'model': models.vgg13}, 'change_model': self.__vgg,
            'vgg13_bn': {'model': models.vgg13_bn}, 'change_model': self.__vgg,
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

        # Import trained model
        model = self.__trained_models[trained_name]['model'](pretrained=True)
        #
        # print( 'Name===>', self.__trained_name,)
        # print( 'model===>', self.__trained_models[self.__trained_name])
        # print( 'change===>', self.__trained_models['change_model'])

        self.__trained_model = self.__trained_models[trained_name]['change_model'](model, fc_layers, out_features)

        # Create the full connected layer
        #self._fc = self.__fc(self.__trained_models[trained_name]['size'], fc_layers, self.dropout, self.out_features)

        # Replace fc in trained model
        # self.__trained_model.fc = self.fc

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

        print('fc==>', fc)

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


# class TrainModel:
#     def __init__(self, model, train_loader, validation_loader, criterion, optimizer, colab_kernel=False, epochs=3, print_every=30):
#         self.model = model
#         self.train_loader = train_loader
#         self.validation_loader = validation_loader
#         self.criterion = criterion
#         self.optimizer = optimizer
#         self.epochs = epochs
#         self.print_every = print_every
#         self.colab_kernel = colab_kernel
#
#     def validation(self, _model, _data_loader, criterion):
#         valid_correct = 0
#         valid_loss = 0
#
#         for images, labels in (iter(_data_loader)):
#             if self.colab_kernel:
#                 images, labels = images.to('cuda'), labels.to('cuda')
#
#             # Forward pass
#             output = _model.forward(images)
#
#             # Track loss
#             valid_loss += loss.item()
#
#
#     steps = 0
#         running_loss = 0
#         for e in range(epochs):
#             # Model in training mode, dropout is on
#             model.train()
#             for images, labels in train_loader:
#                 steps += 1
#
#                 # Flatten images into a 784 long vector
#                 images.resize_(images.size()[0], 784)
#
#                 optimizer.zero_grad()
#
#                 output = model.forward(images)
#                 loss = criterion(output, labels)
#                 loss.backward()
#                 optimizer.step()
#
#                 running_loss += loss.item()
#
#                 if steps % print_every == 0:
#                     # Model in inference mode, dropout is off
#                     model.eval()
#
#                     # Turn off gradients for validation, will speed up inference
#                     with torch.no_grad():
#                         test_loss, accuracy = validation(model, testloader, criterion)
#
#                     print("Epoch: {}/{}.. ".format(e + 1, epochs),
#                           "Training Loss: {:.3f}.. ".format(running_loss / print_every),
#                           "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
#                           "Test Accuracy: {:.3f}".format(accuracy / len(testloader)))
#
#                     running_loss = 0
#
#                     # Make sure dropout and grads are on for training
#                     model.train()


