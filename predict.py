import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data


import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import json
import time
import copy
import argparse

from train import load_model
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str,default = 'flowers/test/101/image_07988.jpg')
parser.add_argument('--checkpoint_path', type=str, default ='model_checkpoint.pth')
parser.add_argument('--topk', type=int,default=5)
parser.add_argument('--label_path', type=str,default='cat_to_name.json')
parser.add_argument('--gpu', action='store_true', default= True)

args, _ = parser.parse_known_args()


if args.img_dir:
    img_dir = args.img_dir

if args.checkpoint_path:
    checkpoint_path=args.checkpoint_path

if args.topk:
    topk = args.topk
    
if args.label_path:
    label_path=args.label_path

if args.gpu:
    gpu=args.gpu
    
def load_checkpoint(checkpoint_path = 'model_checkpoint.pth'):
    checkpoint = torch.load(checkpoint_path)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    model,_,_ = load_model(structure,0.5,hidden_layer1)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    
    img_loader = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456,0.406], std=[0.229,0.224,0.225])
    ])
    out_image = img_loader(pil_image)
    return out_image



def predict(image_dir, model, topk=5,gpu=True):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    if gpu and torch.cuda.is_available():
        model.to('cuda:0')
        image = process_image(image_dir)
        image = image.unsqueeze_(0)
        image = image.float()
    
        with torch.no_grad():
            output = model.forward(image.cuda())
    
        probability = nn.functional.softmax(output.data,dim=1).topk(topk)#.numpy()[0]
        prob= probability[0].data.numpy()[0]
        classes = probability[1].data.numpy()[0]        
        
    else:
        image = process_image(image_dir)
        image = image.unsqueeze_(0)
        image = image.float()
        output = model.forward(image)
        probability = nn.functional.softmax(output.data,dim=1).topk(topk)
        prob= probability[0].data.numpy()[0]
        classes = probability[1].data.numpy()[0]
        

    with open(label_path, 'r') as f:
        cat_to_name = json.load(f)
    labels=list(cat_to_name.values())
    classes = [labels[x] for x in classes]
    
    return prob,classes


model = load_checkpoint(checkpoint_path)
prob, classes = predict(img_dir, model, topk,gpu)

print(prob)
print(classes)


