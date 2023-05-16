from utils import *

def main():
    # load and pre_process data
    data = load_data('./iris.data')
    # initialize weights
    n_clusters = 8  
    weights = initialize_weights(n_clusters, data.shape[1]) # data.shape = (150, 4)
    # train WTA
    weights = train_wta(data, weights, n_epochs=200)
    # test WTA
    labels = test_wta(data, weights)
    # print(labels)
    visualize_3d(data, labels, n_clusters)

if __name__ == '__main__':
    main()
