import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from wikiart import WikiArtDataset, WikiAutoencoder
import matplotlib.patches as mpatches

def load_config(config_file):
    with open(config_file) as f:
        return json.load(f)


def main(config_file):
    config = load_config(config_file)

    # Hyperparameters from config
    DEVICE = config['device'] if torch.cuda.is_available() else 'cpu'
    MODEL_SAVE_PATH = config['modelfile2']  #
    BATCH_SIZE = config.get('batch_size', 32)  

   
    dataset = WikiArtDataset(config['trainingdir'], device=DEVICE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = WikiAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

  
    representations = []

    with torch.no_grad():
        for data in loader:
            img, _ = data
            img = img.to(DEVICE)
            encoded = model.encoder(img).view(img.size(0), -1)
            representations.append(encoded.cpu().numpy())

    representations = np.concatenate(representations)

    # Clustering
    N_CLUSTERS = len(dataset.classes) 
    kmeans = KMeans(n_clusters=N_CLUSTERS)
    clusters = kmeans.fit_predict(representations)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(representations)
    colors = plt.colormaps['tab20'](np.linspace(0, 1, N_CLUSTERS))

 
    plt.figure(figsize=(10,6))
    legend_handles = []
    for cluster_idx in range(N_CLUSTERS):
        cluster_points = reduced[clusters == cluster_idx]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    color=colors[cluster_idx], alpha=0.5)
        
       

    plt.savefig('clusters.png')
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster representations of art styles.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    main(args.config)
