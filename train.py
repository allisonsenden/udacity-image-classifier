# Imports here
import torch
from torch import nn
from torch import optim
from PIL import Image
from torchvision import models

import argparse
from functions import load_data, create_classifier, train_model, check_accuracy_on_test, save_model

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', action='store', type=str, help='Path to data folder.')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available.')
parser.add_argument('--epochs', action='store', type=int, default=3, help='Number of epochs for training the model.')
parser.add_argument('--arch', action='store', type=str, default='vgg16', help='Model architecture: ')
parser.add_argument('--learning_rate', action='store', type=float, default=0.001, help='Learning rate: ')
parser.add_argument('--hidden_layer', action='store', type=int, default=512, help='Number of hidden units in hidden layer.')
parser.add_argument('--dropout', action='store', type=float, default=0.5, help='Dropout rate for training the model.')
parser.add_argument('--checkpoint', action='store', type=str, default='model_checkpoint.pth', help='Save the trained model checkpoint to this filepath.')

results = parser.parse_args()

data_dir = results.data_dir
gpu_mode = results.gpu
epochs = results.epochs
arch = results.arch
learning_rate = results.learning_rate
hidden_layer = results.hidden_layer
dropout = results.dropout
save_dir = results.checkpoint

trainloader, validloader, testloader, train_data, valid_data, test_data = load_data(data_dir)

pretrained_model = results.arch
model = getattr(models, pretrained_model)(pretrained=True)

input_size = model.classifier[0].in_features
create_classifier(model, input_size, hidden_layer, dropout)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

model, optimizer = train_model(model, epochs, trainloader, validloader, criterion, optimizer)
check_accuracy_on_test(model, testloader)


saved_model = save_model(model, train_data, optimizer, epochs, save_dir)
