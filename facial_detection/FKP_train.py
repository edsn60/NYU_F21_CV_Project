import torch
import numpy as np

import FKP_opt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_set, valid_set, loss_function, optimizer, scheduler, type):
    savepath = FKP_opt.saved_model_large_path if type == 0 else FKP_opt.saved_model_small_path
    valid_loss_min = np.Inf  # set initial "min" to infinity

    train_losses = []
    valid_losses = []

    for epoch in range(FKP_opt.epochs):
        train_loss = 0.0

        model.train()
        batch_num = 0
        for batch in train_set:
            optimizer.zero_grad()
            output = model(batch['image'].to(device))
            loss = loss_function(output, batch['keypoints'].to(device))
            loss.backward()

            optimizer.step()
            train_loss += loss.item() * batch['image'].size(0)
            batch_num += 1
            print('Epoch: {}\tBatch: {}\tTraining Loss: {:.6f}'.format(epoch + 1, batch_num, loss))

        valid_loss = 0.0
        model.eval()
        for batch in valid_set:
            output = model(batch['image'].to(device))
            loss = loss_function(output, batch['keypoints'].to(device))
            valid_loss += loss.item() * batch['image'].size(0)

        train_loss = np.sqrt(train_loss / len(train_set.sampler.indices))
        valid_loss = np.sqrt(valid_loss / len(valid_set.sampler.indices))

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'
              .format(epoch + 1, train_loss, valid_loss))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
                  .format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), savepath)
            valid_loss_min = valid_loss

        scheduler.step()

    return train_losses, valid_losses


def predict(model, test_loader):
    model.eval()  # prep model for evaluation

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(batch['image'].to(device)).cpu().numpy()
            if i == 0:
                predictions = output
            else:
                predictions = np.vstack((predictions, output))

    return predictions