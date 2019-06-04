# imports here
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import argparse


data_dir = 'flowers'
train_dir = data_dir + '/train'
train_transforms = transforms.Compose([transforms.Resize(224),
                                       transforms.RandomRotation(30),
                                       transforms.RandomCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
validloader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True)
checkpoint_path = 'model_checkpoint.pth'

def load_data(data_dir):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.Resize(224),
                                       transforms.RandomRotation(30),
                                       transforms.RandomCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(224),
                                           transforms.RandomRotation(30),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)
    validloader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=20)
    
    return trainloader, validloader, testloader, train_data, valid_data, test_data


def process_image(image):
    img_pil = Image.open(image)
    
    adjust = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
    
    tensor = adjust(img_pil)
    np_tensor = np.array(tensor)
    
    return np_tensor


def create_classifier(model, input_size, hidden_layer, dropout):
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(input_size, hidden_layer)),
                                ('relu', nn.ReLU()),
                                ('dropout', nn.Dropout(p=dropout)),
                                ('fc2', nn.Linear(hidden_layer, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                                      ]))

    model.classifier = classifier
    return model


def validation(model, dataset, criterion):
    valid_loss = 0
    accuracy = 0
    
    model.to('cuda')

    for ii, (inputs, labels) in enumerate(validloader):
    
        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy


def train_model(model, epochs, trainloader, validloader, criterion, optimizer):
    steps = 0
    print_every = 30

    model.to('cuda')
    CUDA_LAUNCH_BLOCKING=1

    for e in range(epochs):
        running_loss = 0

        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion)

                print("Epoch: {} ".format(e+1),
                      "Training Loss: {:.3f} ".format(running_loss/print_every),
                      "Validation Loss: {:.3f} ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                model.train()
    return model, optimizer
            
            
def check_accuracy_on_test(model, testloader):    
    correct = 0
    total = 0
    model.to('cuda')
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))

    
def save_model(model, train_data, optimizer, epochs, checkpoint_path):
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': train_data.class_to_idx,
        'classifier': model.classifier,
        'optimizer': optimizer.state_dict,
        'num_epochs': epochs
    }

    saved = torch.save(checkpoint, checkpoint_path)
    return saved
    
    
def load_checkpoint(model, checkpoint_path):
    with open(checkpoint_path, 'rb') as filepath:
        checkpoint = torch.load(filepath)
    
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.classifier = checkpoint['classifier'] 
    model.class_to_idx = checkpoint['class_to_idx']
    return model


def predict(image_path, model_loaded, topk=5):
    model_loaded.to('cuda')
    image = torch.from_numpy(image_path).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    
    
    model_loaded.eval()
    
    with torch.no_grad():
        output = model_loaded.forward(image.cuda())
        
    
    probs = torch.exp(output)
    top_probs = probs.topk(topk)[0]
    top_indices = probs.topk(topk)[1]
    
    top_probs_list = np.array(top_probs)[0]
    top_indices_list = np.array(top_indices[0])
    
    ctx = model_loaded.class_to_idx
    xtc = {x: y for y, x in ctx.items()}
    
    top_classes = []
    for index in top_indices_list:
        top_classes +=[xtc[index]]
    
    return top_probs_list, top_classes