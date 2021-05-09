import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
from tqdm import tqdm
import numpy as np


def train_loop_encoder(model, dataloaders, dataset_sizes, class_weights_labels, alpha=1, beta = 1, custom_loss = 0, num_epochs=50):
    ''' 
    Purpose: This is the training loop function used for training and validation data. We use Early stopping with patience=5.
             We return the model which gives the lowest loss.
    Input:  dataloaders: Train and Validation sets are loaded.
            class_weights_labels: We use class weights to train the model due to unbalanced number of class labels.
            alpha, beta, custom_loss: In case you want to use a custom loss, you can use custom_loss Flag = 1 
                                      and choose alpha& beta values
            num_epochs: We set the maximum number of epochs = 50
    Output: returns the best model (lowest loss)
    '''

    label_loss = nn.NLLLoss(weight=class_weights_labels)
    reconstruction_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-6, eps=1e-08) # clipnorm=1.0, add later

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    val_losses = []
    
    patience = 5 # for early stopping
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode           

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, imgs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                imgs = imgs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, combined_features, decoded_features = model(inputs, imgs)
                    _, preds = torch.max(outputs, 1)
                    actual_labels = torch.max(labels.long(), 1)[1]
                    current_label_loss = label_loss(outputs, actual_labels)
                    current_reconstruction_loss = reconstruction_loss(combined_features, decoded_features)
                    
                    if custom_loss==1:
                        alpha = np.exp(current_label_loss.item())/(np.exp(current_label_loss.item()) + np.exp(current_reconstruction_loss.item()) + 1e-10)
                        beta = 1 - alpha
                    loss = (1+alpha)*current_label_loss + (1+beta)*current_reconstruction_loss

                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == actual_labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'validation':
                val_losses.append(epoch_loss)
            
            # deep copy the model
            if phase == 'validation' and epoch_loss < best_loss:
#                 save_models(epoch,model)
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if phase == 'validation' and epoch >= patience:
                last_losses = val_losses[-patience:]
                if all(x>best_loss for x in last_losses):
                    print("Early stopping...")
                    
                    time_elapsed = time.time() - since
                    print('Training complete in {:.0f}m {:.0f}s'.format(
                        time_elapsed // 60, time_elapsed % 60))
                    print('Best val Loss: {:4f}'.format(best_loss))

                    # load best model weights
                    model.load_state_dict(best_model_wts)
                    return model

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model