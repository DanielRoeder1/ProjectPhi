import torch.nn.functional as F
import torch
from sklearn.manifold import TSNE
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

def get_similarities(tensor):
    """Calculates the mean abs diff and the cosine from each vector to each vector in the input"""
    # Initialize an empty matrix to store the mean differences
    num_embeds= tensor.shape[0]
    mean_diff_matrix = torch.zeros((num_embeds, num_embeds))
    cos_sim_matrix = torch.zeros((num_embeds, num_embeds))
    # Calculate absolute mean differences
    for i in range(num_embeds):
        for j in range(num_embeds):
            # Calculate the absolute difference between vectors
            abs_diff = torch.abs(tensor[i] - tensor[j])
            # Calculate the mean of the absolute differences
            mean_diff = torch.mean(abs_diff)

            sim = F.cosine_similarity(tensor[i].unsqueeze(0), tensor[j].unsqueeze(0))
            # Store the result
            mean_diff_matrix[i, j] = mean_diff
            cos_sim_matrix[i, j] = sim
    return mean_diff_matrix, cos_sim_matrix

def plot_tsne(embeds):
    # t-SNE transformation
    tsne_model = TSNE(n_components=2, random_state=0)
    tsne_embeddings = tsne_model.fit_transform(embeds)

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1])
        
    plt.title('t-SNE of Word Embeddings')
    plt.show()

def plot_pca(embeds):
    pca = PCA(n_components=2)
    reduced_embeddings_pca = pca.fit_transform(embeds)

    # Plot the PCA-reduced embeddings
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings_pca[:, 0], reduced_embeddings_pca[:, 1])
    plt.title('PCA of Word Embeddings')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()