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
import json
import random
import torch.nn.functional as F


class WikiArtImage:
    def __init__(self, imgdir, label, filename):
        self.imgdir = imgdir
        self.label = label
        self.filename = filename
        self.image = None
        self.loaded = False

    def get(self):
        if not self.loaded:
            self.image = read_image(os.path.join(self.imgdir, self.label, self.filename)).float()
            self.loaded = True

        return self.image

class WikiArtDataset(Dataset):
    def __init__(self, imgdir, device="cpu", class_list_file=None, balance_classes = True, style_embedding_dim = 27):
        self.style_embedding_dim = style_embedding_dim
        walking = os.walk(imgdir)
        filedict = {}
        indices = []
        classes = set()
        print("Gathering files for {}".format(imgdir))
        for item in walking:
            #sys.stdout.write('.')
            arttype = os.path.basename(item[0])
            artfiles = item[2]
            for art in artfiles:
                filedict[art] = WikiArtImage(imgdir, arttype, art)
                indices.append(art)
                classes.add(arttype)
        print("...finished")
        if class_list_file and os.path.exists(class_list_file):
            with open(class_list_file, 'r') as f:
                self.classes = json.load(f)
        else:
            self.classes = sorted(list(classes))
            if class_list_file:
                with open(class_list_file, 'w') as f:
                    json.dump(self.classes, f)


        #print("classes (sorted):", self.classes)
        self.filedict = filedict
        self.imgdir = imgdir
        self.indices = indices
        self.device = device

        if balance_classes:
            self._balance_class_data()

    def _balance_class_data(self):
        class_counts = {cls: 0 for cls in self.classes}
        for idx in self.indices:
            label = self.filedict[idx].label
            class_counts[label] += 1
        #print(f"Original class distribution: {class_counts}")

    
        balanced_indices = []
        for cls, count in class_counts.items():
            samples = [idx for idx in self.indices if self.filedict[idx].label == cls]
        
            if count > 300:  
                sampled_indices = random.sample(samples, 300)
            else:  
                sampled_indices = samples * (300 // count) + random.sample(samples, 300 % count)

            balanced_indices.extend(sampled_indices)
    
    
        self.indices = balanced_indices

    
        new_class_counts = {cls: 0 for cls in self.classes}
        for idx in self.indices:
            label = self.filedict[idx].label
            new_class_counts[label] += 1

        #print(f"Class distribution after balancing: {new_class_counts}")


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        imgname = self.indices[idx]
        imgobj = self.filedict[imgname]
        #ilabel = self.classes.index(imgobj.label)
        image = imgobj.get().to(self.device)
        if image.shape != (3, 416, 416):
           transform = torchvision.transforms.Resize((416, 416))
           image = transform(image)
        target = image.clone()
        style_embedding = torch.zeros(self.style_embedding_dim).to(self.device) 
      

        return image, target, style_embedding

class WikiArtModel(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()

        self.conv2d = nn.Conv2d(3, 1, (4,4), padding=2)
        self.maxpool2d = nn.MaxPool2d((4,4), padding=2)
        self.flatten = nn.Flatten()
        self.batchnorm1d = nn.BatchNorm1d(105*105)
        self.linear1 = nn.Linear(105*105, 300)
        self.dropout1 = nn.Dropout(0.01)
        self.relu1 = nn.ReLU()
        self.linear_add = nn.Linear(300, 150)
        self.dropout2 = nn.Dropout(0.01)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(150, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, image):
        output = self.conv2d(image)
        #print("convout {}".format(output.size()))
        output = self.maxpool2d(output)
        #print("poolout {}".format(output.size()))
        output = self.flatten(output)
        output = self.batchnorm1d(output)
        #print("poolout {}".format(output.size()))
        output = self.linear1(output)
        output = self.dropout1(output)
        output = self.relu1(output)
        output = self.linear_add(output)
        output = self.dropout2(output)
        output = self.relu2(output)
        output = self.linear2(output)
        return self.softmax(output)



class WikiAutoencoder(nn.Module):
    def __init__(self):
        super(WikiAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
           nn.Conv2d(3, 3, kernel_size=5, stride=2),  
            nn.BatchNorm2d(3),
            nn.Dropout2d(),
            nn.Conv2d(3, 3, kernel_size=5, stride=2),  
            nn.BatchNorm2d(3),
            nn.Dropout2d(),
            nn.Conv2d(3, 3, kernel_size=5, stride=2),  
            nn.BatchNorm2d(3),
            nn.Dropout2d(),
            nn.Conv2d(3, 1, kernel_size=5, stride=2),  
            nn.BatchNorm2d(1),
            nn.Dropout2d(),

        )   
        # Decoder
        self.decoder = nn.Sequential(
           nn.ConvTranspose2d(1, 3, kernel_size=5, stride=2),  
            nn.BatchNorm2d(3),
            nn.Dropout2d(),
            nn.ConvTranspose2d(3, 3, kernel_size=5, stride=2),  
            nn.BatchNorm2d(3),
            nn.Dropout2d(),
            nn.ConvTranspose2d(3, 3, kernel_size=5, stride=2,  output_padding=1),   
            nn.BatchNorm2d(3),
            nn.Dropout2d(),
            nn.ConvTranspose2d(3, 3, kernel_size=5, stride=2,  output_padding=1),  
            nn.BatchNorm2d(3)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class WikiAutoencoder2(nn.Module):
    def __init__(self, style_embedding_dim=27):
        super(WikiAutoencoder2, self).__init__()
        self.style_embedding_dim = style_embedding_dim

        # Encoder (same as before)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=5, stride=2),
            nn.BatchNorm2d(3),
            nn.Dropout2d(),
            nn.Conv2d(3, 3, kernel_size=5, stride=2),
            nn.BatchNorm2d(3),
            nn.Dropout2d(),
            nn.Conv2d(3, 3, kernel_size=5, stride=2),
            nn.BatchNorm2d(3),
            nn.Dropout2d(),
            nn.Conv2d(3, 1, kernel_size=5, stride=2),
            nn.BatchNorm2d(1),
            nn.Dropout2d(),
        )

       
        self.style_embedding = nn.Linear(style_embedding_dim, 16 * 16)  

        # Decoder (adjusted to take style-augmented input)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 3, kernel_size=5, stride=2, output_padding=1),
            nn.BatchNorm2d(3),
            nn.Dropout2d(),
            nn.ConvTranspose2d(3, 3, kernel_size=5, stride=2, output_padding=1),
            nn.BatchNorm2d(3),
            nn.Dropout2d(),
            nn.ConvTranspose2d(3, 3, kernel_size=5, stride=2, output_padding=1),
            nn.BatchNorm2d(3),
            nn.Dropout2d(),
            nn.ConvTranspose2d(3, 3, kernel_size=5, stride=2, output_padding=1),
            nn.BatchNorm2d(3)
        )

    def forward(self, x, style_embedding):
        encoded = self.encoder(x)
       # print(f"Shape of encoded: {encoded.shape}")
        style_embedding = self.style_embedding(style_embedding).view(-1, 1, 16, 16)
        style_embedding = F.interpolate(style_embedding, size=encoded.shape[-2:], mode='bilinear', align_corners=False)
        #print(f"Shape of style_embedding: {style_embedding.shape}")
        conditioned_encoding = torch.cat([encoded, style_embedding], dim=1)
        #print(f"Shape of conditioned_encoding: {conditioned_encoding.shape}")
        for i, layer in enumerate(self.decoder):
            conditioned_encoding = layer(conditioned_encoding)
            #print(f"Shape after decoder layer {i}: {conditioned_encoding.shape}")
        decoded = F.interpolate(conditioned_encoding, size=(416, 416), mode='bilinear', align_corners=False)
        return decoded
