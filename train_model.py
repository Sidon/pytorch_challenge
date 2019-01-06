import torch

class TrainModel:
    def __init__(self, model, train_loader, validation_loader, criterion, optimizer, epochs=3,
                 gpu_on=False, model_name = None):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.gpu_on = gpu_on
        self.model_name = model_name if model_name is not None else model._getname()

    def train(self, model=None, epochs=None, criterion=None, optimizer=None, train_loader=None, valid_loader=None,
              gpu_on=None):
        model = self.model if model is None else model
        epochs = self.epochs if epochs is None else epochs
        criterion = self.criterion if criterion is None else criterion
        optimizer = self.optimizer if optimizer is None else optimizer
        train_loader = train_loader if train_loader is not None else self.train_loader
        valid_loader = valid_loader if valid_loader is not None else self.validation_loader
        gpu_on = self.gpu_on if gpu_on is None else gpu_on

        min_valid_loss = float('Inf')
        training_loss = 0
        valid_loss = 0
        step = 0

        # move model to cuda
        if gpu_on:
            model.cuda()

        print('Training model: ', self.model_name )
        print('Epochs: ', epochs)
        print('GPU: ', gpu_on)
        print('Model in cuda: ', next(model.parameters()).is_cuda,'\n')

        ## Train the model
        model.train()
        for epoch in range(epochs):

            print('Training, epoch: ', epoch,'/', epochs)
            for images, labels in iter(train_loader):
                step += 1
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

            ## Validate the model
            print('Validation, epoch: ', epoch,'/', epochs)
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

            # save model if validation loss has decreased
            if valid_loss <= min_valid_loss:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    min_valid_loss,
                    valid_loss))
                # SaveCheckpoint()
                torch.save(model.state_dict(), self.model_name+'.pt')
                min_valid_loss = valid_loss
        else:
            pass