import os
import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
import json
from wikiart import WikiArtDataset, WikiAutoencoder2  
import argparse
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser(description="Generate style-conditioned images")
parser.add_argument("--config", type=str, required=True, help="Path to config file")
args = parser.parse_args()

with open(args.config) as f:
    config = json.load(f)

modelfile = config["modelfile3"]
device = config["device"]

testingdir = config["testingdir"]
#style_embedding_dim = config.get("style_embedding_dim", 128)  
testingdataset = WikiArtDataset(config["testingdir"], device=device, class_list_file=config["class_list_file"])  
dataloader = DataLoader(testingdataset, batch_size=config["batch_size"], shuffle=False)


# Load model
model = WikiAutoencoder2(style_embedding_dim=config["style_embedding_dim"]).to(device)
model.load_state_dict(torch.load(modelfile, map_location=device))
model.eval()
model.to(device)

def get_style_embedding(labels):
    return torch.randn(len(labels), config["style_embedding_dim"], device=device)

def generate_images(model, dataloader, output_dir="generated_images", sample_count=1):
    os.makedirs(output_dir, exist_ok=True)  
    
    with torch.no_grad():
       for batch_id, (img, _, _) in enumerate(tqdm(dataloader)):
           if batch_id >= sample_count:
               break

           img = img.to(device)
           style_embedding = get_style_embedding(img)
           generated_img = model(img, style_embedding) 
           
           for i in range(generated_img.shape[0]):
               original_image = img[i].cpu() 
               generated_image = generated_img[i].cpu()  
               original_image_pil = transforms.ToPILImage()(original_image)
               generated_image_pil = transforms.ToPILImage()(generated_image)
               original_image_filename = os.path.join(output_dir, f"original_{batch_id}_{i}.png")
               original_image_pil.save(original_image_filename)
               generated_image_filename = os.path.join(output_dir, f"generated_{batch_id}_{i}.png")
               generated_image_pil.save(generated_image_filename)

generate_images(model, dataloader)
