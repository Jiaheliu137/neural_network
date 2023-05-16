# Project4

## 1.Team member



- **m5268101 Liu** **Jiahe**

- **m5251140 Ken Sato**

- **m5271051 Keisuke** **Utsumi**



## 2.Team Project IV

- Using the Winner-Take-All algorithm to cluster the Iris dataset (http://www.ics.uci.edu/~mlearn/MLRepository.html).
## 3.Mathematical formulas

------

WTA (Winner-Take-All) is a competitive learning algorithm that follows the principle of updating only the winner (the neuron with the highest or lowest output) during each iteration while keeping the other neurons unchanged.

Assuming we have an input vector $\mathbf{x} = (x_1, x_2, ..., x_n)$ and a weight matrix  $\mathbf{W}$，where each row of $\mathbf{W}$ corresponds to the weight vector of a neuron.

### 1.Number of Neurons
The number of neurons is equal to the number of clusters.

### 2.Weight Matrix Initialization
At the beginning of the WTA algorithm, we need to initialize the weight matrix  $\mathbf{W}$.One common approach is to initialize the weights with small random values. This helps the algorithm explore the solution space better in the initial stages.

Here is an example of the weight initialization process:
$$
\begin{align*}
W_{ij} & = \text{rand}(n, m) \\
\end{align*}
$$
$W_{ij}$ represents the element at the $i-th$ row and $j-th$ column of  $\mathbf{W}$ ，$\text{rand}(n, m)$ 表generates a random matrix of size $n$ rows adn  $m$ columns，where $n$ is the number of neurons and ，$m$  is the dimension of the input vector  $\mathbf{x}$.

### 3.Weight Matrix Normalization

Normalize the weight matrix  $\mathbf{W}$, so that each row has a length of 1 (Euclidean norm or L2 norm). The purpose of this step is to ensure that all neuron weights are in the same order of magnitude in the initial state, avoiding learning instability caused by some weights being too large or too small.

Below is the process of weight matrix normalization:
$$
\begin{align*}
\mathbf{W}_{i,:} & = \frac{\mathbf{W}_{i,:}}{\|\mathbf{W}_{i,:}\|} \\
\end{align*}
$$
$\|\mathbf{W}*{i,:}\|$  represents the length (or norm) of the weight vector $\mathbf{W}*{i,:}$
For an n-dimensional vector $x$ , its Euclidean norm can be represented as:
$$
\begin{equation}
\|\mathbf{x}\|_2 = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2} \quad \text or \quad \|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2} 
\end{equation}
$$
This process involves operating on each row of the weight matrix $\mathbf{W}$. $\|\mathbf{W}\|_2$ denotes the calculation of the length of each row of the weight matrix $\mathbf{W}$, resulting in a vector.
$$
\begin{align*}
\mathbf{W} & = \frac{\mathbf{W}}{\|\mathbf{W}\|_2} \\
\end{align*}
$$
### 4.Updating the Weight Matrix
In each iteration, we need to compute the dot product of the input vector $\mathbf{x}$ and each row of the weight matrix $\mathbf{W}$, and then select the neuron with the largest dot product for update.

Here is an example of the weight matrix update process:
$$
\begin{align*}
i^* & = \arg\max_i \mathbf{x} \cdot \mathbf{W}_{i,:} \\
\mathbf{W}_{i^*,:} & = \mathbf{W}_{i^*,:} + \alpha (\mathbf{x} - \mathbf{W}_{i^*,:}) \\
\end{align*}
$$
$i^*$ represents the index of the neuron with the largest dot product. Mathematically, the largest dot product represents the most consistent direction, which means the weight and the input pattern are most similar.

$\mathbf{W}_{i,:}$ represents the $i$-th row of the weight matrix $\mathbf{W}$, and $\alpha$ is the learning rate, which controls the step size of the weight update.

The Winner-Take-All algorithm only updates the weights of the neuron corresponding to $i^*$, while the weights of other neurons remain unchanged.

It's important to note that before updating the matrix, the training set needs to be normalized to make the range of its elements consistent with the range of the elements in the weight matrix. After updating the matrix, continue to use the L2 norm to normalize the weight matrix again.

When preprocessing the dataset, columns need to be normalized because the same feature exists across a column.



### 5.The strusture of Iris dataset

```python
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5.0,3.6,1.4,0.2,Iris-setosa
5.4,3.9,1.7,0.4,Iris-setosa
4.6,3.4,1.4,0.3,Iris-setosa
5.0,3.4,1.5,0.2,Iris-setosa
4.4,2.9,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
5.4,3.7,1.5,0.2,Iris-setosa
4.8,3.4,1.6,0.2,Iris-setosa
4.8,3.0,1.4,0.1,Iris-setosa
4.3,3.0,1.1,0.1,Iris-setosa
....
....
```



## 4.Implement code

[project4.py](./project4.py)

```python
def main():
    # load and pre_process data
    data = load_data('./iris.data')
    # initialize weights
    n_clusters = 8  
    weights = initialize_weights(n_clusters, data.shape[1]) # data.shape = (150, 4)
    # train WTA
    weights = train_wta(data, weights, n_epochs=400)
    # test WTA
    labels = test_wta(data, weights)
    # print(labels)
    visualize_3d(data, labels, n_clusters)

if __name__ == '__main__':
    main()

```

[utils.py](./utils.py)

```python
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
    cmap_list = ['cool', 'hot', 'jet', 'viridis', 'rainbow', 'gray']
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

```



## 5.Results

Try setting different numbers of clusters to observe the effect of the neural network. When the number of clusters is small, a two-dimensional image is sufficient. However, when the number of clusters is large, different classes may overlap in two-dimensional space. In such cases, use a three-dimensional image.

### 1.n_clusters = 2

![截屏2023-05-16 12.36.28](./Report_project4_EN.assets/%E6%88%AA%E5%B1%8F2023-05-16%2012.36.28.png)

### 2.n_clusters = 3

![截屏2023-05-16 12.37.52](./Report_project4_EN.assets/%E6%88%AA%E5%B1%8F2023-05-16%2012.37.52.png)

### 3.n_clusters = 4

![截屏2023-05-16 12.39.10](./Report_project4_EN.assets/%E6%88%AA%E5%B1%8F2023-05-16%2012.39.10.png)

### 4.n_clusters = 5

![截屏2023-05-16 12.40.19](./Report_project4_EN.assets/%E6%88%AA%E5%B1%8F2023-05-16%2012.40.19.png)

### 5.n_clusters = 6

![截屏2023-05-16 12.42.42](./Report_project4_EN.assets/%E6%88%AA%E5%B1%8F2023-05-16%2012.42.42.png)

### 6.n_clusters = 7

![截屏2023-05-16 12.45.07](./Report_project4_EN.assets/%E6%88%AA%E5%B1%8F2023-05-16%2012.45.07.png)

### 7.n_clusters = 8

![截屏2023-05-16 12.47.03](./Report_project4_EN.assets/%E6%88%AA%E5%B1%8F2023-05-16%2012.47.03.png)

### 8.Conclusion

From the above images, it can be seen that for the Iris dataset, the WTA (Winner-Take-All) neural network has a good clustering capability. It's also noticed that even as the number of 'n_clusters' increases, the actual number of types of classes doesn't necessarily increase. That is, the real number of clusters in the data may be less than the number of clusters we set. This indicates that the number of intrinsic distribution patterns in the Iris dataset itself is limited. Judging intuitively from the above images, the number of classes determined by the intrinsic data structure of Iris is probably between 2-5.
