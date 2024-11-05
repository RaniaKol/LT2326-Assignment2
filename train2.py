import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.optim import Adam
import tqdm
from wikiart import WikiArtDataset, WikiAutoencoder
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file", default="config.json")
args = parser.parse_args()
config = json.load(open(args.config))


trainingdir = config["trainingdir"]
device = config["device"]
modelfile = config["modelfile2"]
epochs = config["epochs"]
batch_size = config["batch_size"]
class_list_file = config.get("class_list_file", "class_indices.json")
balance_classes = config.get("balance_classes", True)
print("Running...")


traindataset = WikiArtDataset(trainingdir, device)
loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)


def train2(epochs=3, batch_size=32, modelfile=None, device="cpu"):
    model = WikiAutoencoder().to(device)  
    optimizer = Adam(model.parameters(), lr=0.01) 
    criterion = nn.MSELoss()  

    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}/{epochs}")
        accumulate_loss = 0
        for batch_id, (X, _) in enumerate(tqdm.tqdm(loader)):
            X = X.to(device)  
            optimizer.zero_grad()  
            output = model(X)  
            loss = criterion(output, X)  
            loss.backward()  
            optimizer.step()  
            accumulate_loss += loss.item()  

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {accumulate_loss / len(loader):.4f}") 

    if modelfile:
        torch.save(model.state_dict(), modelfile)  

    return model

model = train2(config["epochs"], config["batch_size"], modelfile=config["modelfile2"], device=device)
