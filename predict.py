# Predicting classes
# - The predict.py script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability

# Top K classes
# - The predict.py script allows users to print out the top K classes along with associated probabilities

# Displaying class names
# - The predict.py script allows users to load a JSON file that maps the class values to other category names

# Predicting with GPU
# - The predict.py script allows users to use the GPU to calculate the predictions



# Imports here
# %matplotlib inline
# import matplotlib.pyplot as plt

import time
import numpy as np
from PIL import Image
import json
import argparse

import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


# ----------------------------------------------------------------------
# COMMAND LINE ARGUMENTS
# parser.add_argument('--', type= , default= , help= )
def get_parse_args():
    parser = argparse.ArgumentParser()
    
    # Predicting classes - reads in an image and a checkpoint then prints the most likely image class and it's associated probability
    parser.add_argument('--image_path', type=str, default='flowers/test/90/image_04428.jpg', help='path to image')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='path to checkpoint')
    
    # Top K classes - to print out the top K classes along with associated probabilities
    parser.add_argument('--topk', type=int, default=5, help='top k classes')
    
    # Displaying class names - to load a JSON file that maps the class values to other category names
    parser.add_argument('--cat_name', type=str, default='cat_to_name.json', help='file that maps class values to category name')
    
    # Predicting with GPU - to use the GPU to calculate the predictions
    parser.add_argument('--gpu', type=str, help='using gpu or cpu')
    
    return parser.parse_args()

# ----------------------------------------------------------------------
# - Loading checkpoints
#   : There is a function that successfully loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)    
    
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)  
   
    model.classifier = checkpoint['classifier']  
    model.epochs = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])   
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer = (checkpoint['optimizer'])
    
    return model

# ----------------------------------------------------------------------
# - Image Processing
#   : The process_image function successfully converts a PIL image 
#     into an object that can be used as input to a trained model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # Use PIL to load the image
    pil_image = Image.open(image)

    # resize the images where the shortest side is 256 pixels
    pil_image.thumbnail((256, 256))
    
    # crop out the center 224x224 portion
    l = (pil_image.width - 224) / 2
    t = (pil_image.height - 224) / 2
    r = l + 224
    b = t + 224
    pil_image = pil_image.crop((l, t, r, b))  
        
    # Convert color cannels to Numpy array. use np_image = np.array(pil_image) and normalize
    np_image = np.array(pil_image) / 255.0

    # means = [0.485, 0.456, 0.406] and standard deviations = [0.229, 0.224, 0.225]
    # subtract the means from each color channel, then divide by the standard deviation
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stds

    # Reorder dimensions according to PyTorch using ndarray.transpose. 
    # The color channel needs to be first and retain the order of the other two dimensions.  
    np_image = np_image.transpose((2, 0, 1))   

    return np_image  


# ----------------------------------------------------------------------
# - Class Prediction
#   : The predict function successfully takes the path to an image and a checkpoint, 
#     then returns the top K most probably classes for that image
def predict(image_path, model, topk, dev):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    
    image = process_image(image_path)
    # image = torch.from_numpy(np.array([image])).float()  # image -> numpy -> tensor (datatype to float)
 
    if dev == 'gpu':
        model = model.gpu();
        image  = torch.from_numpy(image).type(torch.cuda.FloatTensor)
        image = image.unsqueeze(0) 
        image = image.cuda();
    else:
        model = model.cpu()
        image  = torch.from_numpy(image).type(torch.FloatTensor)
        image = image.unsqueeze(0) 
                           
    # image = Variable(image)

    with torch.no_grad():
        output = model.forward(image)
    
    # Get the top k probabilities and their indices
    probs, indices = torch.topk(output, topk)
    probs = np.exp(probs.numpy()[0]) 
    indices = indices.numpy()[0]
        
    # Convert indices to class labels using class_to_idx mapping
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[i] for i in indices]

    return probs, classes    


# ----------------------------------------------------------------------
def main():
    args = get_parse_args()
    
    if args.gpu == 'gpu':
        device = torch.device("cuda")
        dev = 'gpu'
    else:
        print('GPU is not available')
        device = torch.device("cpu")
        dev = 'cpu'
    
    model = load_checkpoint(args.checkpoint)
    
    with open(args.cat_name, 'r') as f:
        cat_to_name = json.load(f)

    image_path = args.image_path
    probs, classes = predict(image_path, model, args.topk, dev)
    ori_lbl = classes[np.argmax(probs)]
    label = [cat_to_name[i] for i in classes]

    print("\n")
    print(image_path)
    print(f"Original - [{ori_lbl}]: {cat_to_name[ori_lbl]} with probability of {probs[np.argmax(probs)]:.4f}\n")
    # print(probs)
    # print(classes)

    print("Top 5 Classes with Probability")
    for id in range (args.topk):
        print(f"{id+1} - [{classes[id]}]: {label[id]} (probabilty = {probs[id]:.4f})")
        
# ----------------------------------------------------------------------
if __name__ == '__main__':
    main()

