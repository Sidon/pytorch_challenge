====================
Structure of Project
====================

torch_helper.py
---------------

**This module defines the following classes**


    `MyNet <https://github.com/Sidon/pytorch_challenge/blob/89c785b420d7557708a795900e1dd25b9da4f234/torch_helper.py#L15>`_
      Class to the definition of the model

      Methods
        To change the last layer:
             | __vgg(self, model, fc=None, out_features=102)
             | __densenet(self, model, fc=None, out_features=102)
             | __inception(self, model, fc=None, out_features=102)
             | __squeeznet(self, model, fc=None, out_features=102)
             | __resnet(self, model, fc=None, out_features=102)

        To creating optimizer
            create_optmizer(self, optimizer=None, parameters=None, \**kwargs)


    **Transforms**
        Class for definition of transforms

    **Loaders**
        Class for the definition of loaders


train_model.py
--------------

**This module defines the following classes**

    `TrainModel <https://github.com/Sidon/pytorch_challenge/blob/89c785b420d7557708a795900e1dd25b9da4f234/train_model.py#L3>`_
      Class to definition of training.

      Methods
        To the definition of the training atributs and to make looping in epochs
              | The main method of this class
              | `train(self, model=None, epochs=None, ...gpu_on=False) <https://github.com/Sidon/pytorch_challenge/blob/89c785b420d7557708a795900e1dd25b9da4f234/train_model.py#L56>`_

        To steps, backwards and calculate loss
              | This method is called by the train method.
              | `__train(self, model, criterion, optimizer, batch, gpu_on) <https://github.com/Sidon/pytorch_challenge/blob/89c785b420d7557708a795900e1dd25b9da4f234/train_model.py#L16>`_

        To validation
              | This method is called by the train method.
              | `__validation(self, model, criterion, batch, gpu_on) <https://github.com/Sidon/pytorch_challenge/blob/89c785b420d7557708a795900e1dd25b9da4f234/train_model.py#L43>`_