import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms.functional as F
from torch.optim import Adam
import tqdm
from wikiart import WikiArtDataset, WikiAutoencoder2  
import json
import argparse
import numpy as np
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file", default="config.json")

args = parser.parse_args()

config = json.load(open(args.config))

trainingdir = config["trainingdir"]
testingdir = config["testingdir"]
device = config["device"]
class_list_file = config.get("class_list_file", "class_indices.json")
balance_classes = config.get("balance_classes", True)
style_embedding_dim = 27 

print("Running...")


traindataset = WikiArtDataset(trainingdir, device, class_list_file, style_embedding_dim=27)
print(f"Image directory: {traindataset.imgdir}")

the_image, the_label, style_embedding = traindataset[5]
print(f"Image label: {the_label}, Image size: {the_image.size()}")

def get_style_embedding(labels):
    return torch.randn(len(labels), style_embedding_dim, device=device)  

def train(epochs=3, batch_size=32, modelfile=None, device="cpu"):
    print(f"Length of training dataset: {len(traindataset)}")
    loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)  
    model = WikiAutoencoder2(style_embedding_dim).to(device)  
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss().to(device)

    for epoch in range(epochs):
        print(f"Starting epoch {epoch}")
        accumulate_loss = 0

        for batch_id, batch in enumerate(tqdm.tqdm(loader)):
            X, y, style_embedding = batch  
            X = X.to(device)  
            y = y.to(device)
            style_embedding = style_embedding.to(device)  
            optimizer.zero_grad()
            output = model(X, style_embedding)
            #print(f"Shape of output: {output.shape}") 
            #print(f"Shape of target (y): {y.shape}")
            if output.shape != y.shape:
               raise ValueError(f"Expected y to be {output.shape}, got {y.shape}")
            loss = criterion(output, y)  
            loss.backward()
            accumulate_loss += loss.item()
            optimizer.step()

        print(f"In epoch {epoch}, loss = {accumulate_loss:.4f}")

    if modelfile:
        torch.save(model.state_dict(), modelfile)

    return model

model = train(config["epochs"], config["batch_size"], modelfile=config["modelfile3"], device=device)
