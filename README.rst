====================
Structure of Project
====================

torch_helper.py
---------------

**This module defines the following classes**


    **MyNet**
      Class to the definition of the model

      Methods
        To change the last layer:
             __vgg(self, model, fc=None, out_features=102)

             __densenet(self, model, fc=None, out_features=102)

             __inception(self, model, fc=None, out_features=102)

             __squeeznet(self, model, fc=None, out_features=102)

             __resnet(self, model, fc=None, out_features=102)

        To creating optimizer
            create_optmizer(self, optimizer=None, parameters=None, \**kwargs)


**Transforms**
  Class for definition of transforms

**Loaders**
  Class for the definition of loaders


train_model.py
--------------

**This module defines the following classes**

|    **TrainModel**
|      Class to definition of training.
|
|      Methods
|        To the definition of the training atributs and to make looping in epochs|
|              train(self, model=None, epochs=None, ...gpu_on=False):
|
|        To steps, backwards and calculate loss
|              __train(self, model, criterion, optimizer, batch, gpu_on)
|
|        To validation|
|            __validation(self, model, criterion, batch, gpu_on)
|
