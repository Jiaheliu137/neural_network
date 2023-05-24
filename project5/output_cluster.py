import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import random
import argparse

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
        labels.append("class"+str(winner))
    return labels

def cluster_summary(data, labels, n_clusters):
    cluster_counts = []
    for i in range(n_clusters):
        cluster_data = data[np.array(labels) == i]
        count = len(cluster_data)
        cluster_counts.append(count)
        print(f'Pattern {i} includes {count} data points.')
    return cluster_counts


def main(n_clusters):
    # load and pre_process data
    data = load_data('./iris.data')
    # initialize weights
    weights = initialize_weights(n_clusters, data.shape[1]) # data.shape = (150, 4)
    # train WTA
    weights = train_wta(data, weights, n_epochs=200)
    # test WTA
    labels = test_wta(data, weights)
    # print(labels)
    with open("clusters_"+str(n_clusters)+".txt", 'w') as file:
        for item in labels:
            file.write(str(item) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cluster", help="number of clusters", default=3, type=int)
    args = parser.parse_args()
    main(args.cluster)
