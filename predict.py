# imports here
import torch
import json
import argparse
from torchvision import datasets, transforms, models
from PIL import Image

from functions import process_image, load_checkpoint, predict

checkpoint_path = 'model_checkpoint.pth'

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', action='store', default = 'flowers/test/10/image_07104.jpg', dest='image_path',
                    help='Path to an image to test the model on.')

parser.add_argument('--arch', action='store', default='vgg16', dest='arch',
                    help='Pretrained model you would like to use, default is VGG-16')

parser.add_argument('--cat_to_name', action='store', default = 'cat_to_name.json', dest='cat_to_name',
                    help='Path to an image.')

parser.add_argument('--gpu', action="store_true", default=True, dest='gpu',
                    help='GPU model on or off.')

parser.add_argument('--top_k', action='store', default = 5, dest='top_k',
                    help='The number of predictions returned.')

results = parser.parse_args()

image = results.image_path
gpu_mode = results.gpu
category_names = results.cat_to_name
pretrained_model = results.arch
top_k = results.top_k

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

model = getattr(models, pretrained_model)(pretrained=True)
model_loaded = load_checkpoint(model, checkpoint_path)

image_processed = process_image(image)


probs, classes = predict(image_processed, model_loaded, top_k)

print(probs)
print(classes) 

names = []
for i in classes:
    names += [cat_to_name[i]]
final_percent = round(probs[0]*100, 4)
print("This flower is predicted to be a {} with a probability of {}% ".format(names[0], final_percent))
