import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data


import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import time
import copy
import argparse

# Define command line arguments

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', action="store",type=str, default="./flowers/")
parser.add_argument('--gpu', action="store",type=str, default=True)
parser.add_argument('--learning_rate', action = "store", default = 0.001, type=float)
parser.add_argument('--dropout',action = "store", default=0.5)
parser.add_argument('--epochs',action = "store", type = int, default = 1)
parser.add_argument('--structure', action = "store", type=str, default = "vgg16")
parser.add_argument('--hidden_layer', action="store", type=int, default = 120)

args, _ = parser.parse_known_args()

if args.data_dir:
    data_dir = args.data_dir

if args.gpu:
    gpu=args.gpu

if args.learning_rate: 
    lr=args.learning_rate

if args.dropout:
    drop_out = args.dropout

if args.epochs:
    epochs=args.epochs

if args.structure:
    structure = args.structure

if args.hidden_layer:
    hidden_layer1 = args.hidden_layer

arch = {"vgg16":25088,
        "densenet121":1024,
        "alexnet":9216}

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = {
    'train':transforms.Compose([
        transforms.RandomRotation(50),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]),
    'valid':transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]),
    'test':transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
}

# TODO: Load the datasets with ImageFolder
image_datasets = {
    'train':datasets.ImageFolder(train_dir,transform = data_transforms['train']),
    'valid':datasets.ImageFolder(valid_dir, transform = data_transforms['valid']),
    'test':datasets.ImageFolder(test_dir, transform = data_transforms['test'])
}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
    'train':data.DataLoader(image_datasets['train'],batch_size=64, shuffle=True),
    'valid':data.DataLoader(image_datasets['valid'],batch_size=32,shuffle=True),
    'test':data.DataLoader(image_datasets['test'],batch_size=20,shuffle=True)
}

def load_model(structure = 'vgg16', drop_out=0.5, hidden_layer1=4096, lr=0.001,gpu=True):
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("Im sorry but {} is not a valid model.Did you mean vgg16,densenet121,or alexnet?".format(structure))

    for param in model.parameters():
        param.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(drop_out)),
            ('inputs', nn.Linear(arch[structure], hidden_layer1)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hidden_layer1, 4096)),
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(4096,4096)),
            ('relu3',nn.ReLU()),
            ('hidden_layer3',nn.Linear(4096,102)),
            ('output', nn.LogSoftmax(dim=1))
                          ]))


        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr )

        if gpu and torch.cuda.is_available():
            model.cuda()

        return model, criterion, optimizer

    
    
def train_model(model,criterion,optimizer, epochs=10, gpu=True):
    steps = 0
    running_loss = 0
    loader=dataloaders['train']
    print("--------------Training is starting------------- ")
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(loader):
            steps += 1
            if gpu and torch.cuda.is_available():
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                
                optimizer.zero_grad()
                
                # Forward and backward passes
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if steps % 3 == 0:
                    model.eval()
                    val_loss = 0
                    accuracy=0
                    
                    for ii, (inputs2,labels2) in enumerate(dataloaders['valid']):
                        optimizer.zero_grad()
                        if gpu and torch.cuda.is_available():
                            inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                            model.to('cuda:0')
                            
                            with torch.no_grad():
                                outputs = model.forward(inputs2)
                                val_loss = criterion(outputs,labels2)
                                ps = torch.exp(outputs).data
                                equality = (labels2.data == ps.max(1)[1])
                                accuracy += equality.type_as(torch.FloatTensor()).mean()
                                
                    val_loss = val_loss / len(dataloaders['valid'])
                    accuracy = accuracy /len(dataloaders['valid'])
                    
                    print("Epoch: {}/{}... ".format(e+1, epochs),
                          "Train Loss: {:.4f}".format(running_loss/3),
                          "Valid Loss: {:.4f}".format(val_loss),
                          "Accuracy: {:.4f}".format(accuracy))
                    
                    running_loss = 0
    
    print("Epochs: {} , Steps: {}".format(epochs,steps))
    print("-------------- Finished training -----------------------")
    

model, criterion, optimizer=load_model(structure, drop_out,hidden_layer1, lr, gpu)#(structure = 'alexnet', drop_out=0.5, hidden_layer1=4096, lr=0.001,gpu=True)    
train_model(model,criterion,optimizer, epochs, gpu)#(model,criterion,optimizer, epochs=10, loader=dataloaders['train'], gpu=True)
model.class_to_idx = image_datasets['train'].class_to_idx
torch.save({'structure' :structure,
            'hidden_layer1':hidden_layer1,
            'state_dict':model.state_dict(),
            'class_to_idx':model.class_to_idx},
            'model_checkpoint.pth')