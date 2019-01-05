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

    def train(self, model=None, epochs=None, criterion=None, optimizer=None, train_loader=None, valid_loader=None,
              gpu_on=False):
        model = self.model if model is None else model
        epochs = self.epochs if epochs is None else epochs
        criterion = self.criterion if criterion is None else criterion
        optimizer = self.optimizer if optimizer is None else optimizer
        train_loader = train_loader if train_loader is not None else self.train_loader
        valid_loader = valid_loader if valid_loader is not None else self.validation_loader

        min_valid_loss = float('Inf')
        training_loss = 0
        valid_loss = 0

        ## Train the model
        model.train()
        for epoch in range(epochs):
            for images, labels in iter(train_loader):

                # move data to gpu
                if (gpu_on):
                    images, labels = images.cuda(), labels.cuda()

                # Clear the gradients, do this because gradients are accumulated
                optimizer.zero_grad()
                # Forward pass, then backward pass, then update weights
                output = model.forward(images)
                # calculate the batch loss
                loss = criterion(output, labels)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update training loss
                training_loss += loss.item()

                print('epoch: ', epoch, 'training_loss: ', training_loss)

            ## Validate the movel
            model.eval()
            for images, labels in iter(valid_loader):
                # move data to gpu
                if (gpu_on):
                    images, labels = images.cuda(), labels.cuda()

                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(images)
                # calculate the batch loss
                loss = criterion(output, labels)
                # update average validation loss
                valid_loss += loss.item() * images.size(0)

            # calculate average losses
            training_loss = training_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(valid_loader.dataset)

            # print training/validation statistics
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, training_loss, valid_loss))

            # save model if validation loss has decreased
            if valid_loss <= min_valid_loss:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    min_valid_loss,
                    valid_loss))
                # SaveCheckpoint()
                torch.save(model.state_dict(), 'model_part.pt')
                min_valid_loss = valid_loss
        else:
            pass