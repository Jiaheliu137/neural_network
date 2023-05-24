import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import random

def load_data(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
    dataset = [row for row in dataset if len(row) == 5]
    for i in range(len(dataset)):
        dataset[i] = dataset[i][:-1]
    dataset = np.array(dataset, dtype=float)

    # normalization
    min_values = dataset.min(axis=0)
    max_values = dataset.max(axis=0)
    norm_dataset = (dataset - min_values) / (max_values - min_values) - 0.5

    return norm_dataset



def initialize_weights(n_clusters, n_features):
    weights = np.random.rand(n_clusters, n_features) - 0.5
    # normalization
    norm = np.linalg.norm(weights, axis=1, keepdims=True) # norm.shape = (3, 1)
    weights /= norm # weights.shape = (3, 4)
    return weights

# train WTA
def train_wta(data, weights, alpha=0.5, n_epochs=20):
    for epoch in range(n_epochs):
        for x in data:
            outputs = np.dot(weights, x) # (3, 4) inner product (4, ),outputs.shape = (3, )
            winner = np.argmax(outputs)
            # update weights
            weights[winner] += alpha * (x - weights[winner])
            # normalization
            weights[winner] /= np.linalg.norm(weights[winner])
    return weights

# test WTA，and return labels of each pattern
def test_wta(data, weights):
    labels = []
    for x in data:
        outputs = np.dot(weights, x)
        winner = np.argmax(outputs)
        labels.append(winner)
    return labels

def cluster_summary(data, labels, n_clusters):
    cluster_counts = []
    for i in range(n_clusters):
        cluster_data = data[np.array(labels) == i]
        count = len(cluster_data)
        cluster_counts.append(count)
        print(f'Pattern {i} includes {count} data points.')
    return cluster_counts

# visualization
def visualize_2d(data, labels, n_clusters):
    plt.figure()
    cmap_list = ['cool', 'hot', 'jet', 'viridis', 'rainbow', 'gray']
    cmap = plt.cm.get_cmap(random.choice(cmap_list), n_clusters)  # get color map
    cluster_counts = cluster_summary(data, labels, n_clusters)
    for i in range(n_clusters):
        cluster_data = data[np.array(labels) == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=cmap(i), label=f'pattern{i} - count: {cluster_counts[i]}')
    plt.legend(loc='upper right')
    plt.xlim(-0.6, 0.8)
    plt.ylim(-0.6, 0.8)
    plt.title('Number of Clusters: {}'.format(n_clusters))
    plt.show()


def visualize_3d(data, labels, n_clusters):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #cmap_list = ['cool', 'hot', 'jet', 'viridis', 'rainbow', 'gray']
    cmap_list = ['rainbow']
    cmap = plt.cm.get_cmap(random.choice(cmap_list), n_clusters)  # get color map
    cluster_counts = cluster_summary(data, labels, n_clusters)
    for i in range(n_clusters):
        cluster_data = data[np.array(labels) == i]
        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], color=cmap(i), label=f'pattern{i}: {cluster_counts[i]}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-0.5, 0.6)
    ax.set_ylim(-0.5, 0.6)
    ax.set_zlim(-0.5, 0.6)
    ax.legend(loc='upper left', bbox_to_anchor=(1,1)) # set the uppder left cornor in (1, 1)
    ax.set_title('Number of Clusters: {}'.format(n_clusters))
    plt.show()
