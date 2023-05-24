from utils import *
import argparse

def main(n_clusters, visualize):
    # load and pre_process data
    data = load_data('./iris.data')
    # initialize weights
    weights = initialize_weights(n_clusters, data.shape[1]) # data.shape = (150, 4)
    # train WTA
    weights = train_wta(data, weights, n_epochs=200)
    # test WTA
    labels = test_wta(data, weights)
    print(labels)
    if visualize == 2:
        visualize_2d(data, labels, n_clusters)
    if visualize == 3:
        visualize_3d(data, labels, n_clusters)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visualize", help="visualize option", type=int, default=2)
    parser.add_argument("-c", "--cluster", help="cluster number", type=int, default=3)
    args = parser.parse_args()
    main(args.cluster, args.visualize)
