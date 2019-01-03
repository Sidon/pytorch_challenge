import torch

class TrainModel:
    def __init__(self, model, train_loader, validation_loader, criterion, optimizer, colab_kernel=False, epochs=3,
                 gpu_on=False, model_name = None):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.print_every = print_every
        self.colab_kernel = colab_kernel
        self.gpu_on = gpu_on
        self.model_name = model_name if model_name is not None else model._getname()


    def __train(self, model, criterion, optimizer, loader, gpu_on):

        running_loss = 0
        for image, label in next(loader):
            # move data to gpu
            if (gpu_on):
                image, label = image.cuda(), label.cuda()

            # Clear the gradients, do this because gradients are accumulated
            optimizer.zero_grad()

            # Forward pass, then backward pass, then update weights
            output = model.forward(image)

            # calculate the batch loss
            loss = criterion(output, label)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()

            # update training loss
            running_loss += loss.item()
        else:
            return running_loss


    def __validation(self, model, criterion, loader, gpu_on):
        running_loss = 0
        for data, target in loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # update running validation loss
            running_loss += loss.item() * data.size(0)
        else:
            return running_loss


    def train(self, model=None, epochs=None, criterion=None, optimizer=None, train_loader=None, valid_loader=None,
              gpu_on=False):
        model = self.model if model is None else model
        epochs = self.epochs if epochs is None else epochs
        criterion = self.criterion if criterion is None else criterion
        optimizer = self.optimizer if optimizer is None else optimizer
        train_loader = train_loader if train_loader is not None else self.train_loader
        valid_loader = valid_loader if valid_loader is not None else self.validation_loader

        min_valid_loss = float('Inf')

        for epoch in range(epochs):

            # Model in training mode, dropout is on
            model.train()
            train_loss = self.__train(model, criterion, optimizer, train_loader, gpu_on)

            # Model in validation mode
            model.eval()
            valid_loss = self.__validation(model, criterion, optimizer, valid_loader, gpu_on)

            # save model if validation loss has decreased
            if valid_loss <= min_valid_loss:
                print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model... '.format(valid_loss, min_valid_loss))
                torch.save(model.state_dict(), self.model_name )
                min_valid_loss = valid_loss
        else:
            pass




