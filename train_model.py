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
        self.Traindeficolab_kernel = colab_kernel
        self.gpu_on = gpu_on
        self.model_name = model_name if model_name is not None else model._getname()


    def __train(self, model, criterion, optimizer, batch, gpu_on):
       running_loss = 0
       for image, label in batch:
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

    def __validation(self, model, criterion, batch, gpu_on):
        running_loss = 0
        for data, target in batch:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model.forward(data)
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
        iter_train_loader = iter(train_loader)
        iter_valid_loader = iter(valid_loader)

        for epoch in range(epochs):

            # Model in training mode, dropout is on
            model.train()
            batch_train = next(iter_train_loader)
            batch_valid = next(iter_valid_loader)
            z_batch_train = zip(batch_train[0], batch_train[1])
            z_batch_valid = zip(batch_valid[0], batch_valid[1])

            train_loss = self.__train(model, criterion, optimizer, z_batch_train, gpu_on)

            # Model in validation mode
            model.eval()
            valid_loss = self.__validation(model, criterion, optimizer, z_batch_valid, gpu_on)

            # save model if validation loss has decreased
            if valid_loss <= min_valid_loss:
                print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model... '.format(valid_loss, min_valid_loss))
                torch.save(model.state_dict(), self.model_name )
                min_valid_loss = valid_loss
        else:
            pass