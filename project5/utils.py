import numpy as np
import random
from math import sqrt, pow
import matplotlib.pyplot as plt
import matplotlib
from itertools import product


M1 = 25
M2 = 25
N = M1 * M2




def InputPattern(filename):
    data = np.genfromtxt(filename, delimiter=',', dtype=None, encoding=None)
    P = data.shape[0]  # number of patterns
    I = len(data[0]) - 1  # input dimension
    x = np.zeros((P, I))
    classes = np.array([d[-1] for d in data])  # get the class labels
    # print(classes)
    label0 = np.arange(P)  # use sequence number as label
    label_dict = {label0[i]: classes[i] for i in range(P)}  # dictionary to remember the correspondence between labels and classes
    
    # Get unique classes and create color_dict
    unique_classes = np.unique(classes)
    color_list = ['red', 'blue', 'green', 'purple', 'cyan', 'orange', 'magenta', 'lime', 'navy', 'darkred', 'darkgreen', 'gold', 'teal']
    random.shuffle(color_list)  # Randomly shuffle the color list
    color_dict = {unique_classes[i]: color_list[i] for i in range(len(unique_classes))}
    # print(color_dict)    
    
    # Convert the structured array to a regular numpy array
    data_array = np.array([list(d)[:I] for d in data])
    x = data_array
    for p in range(P):
        for i in range(I):
            print(x[p][i], end='  ')
        print(label0[p])

    return x, label0, label_dict, I, P, color_dict


def SOFM(n_update, r1, r2, Nc1, Nc2, w, x, I, P):
    for q in range(n_update):
        p = random.randint(0, P - 1)
        d0 = float('inf')
        for m1 in range(M1):
            for m2 in range(M2):
                d = sum([pow(w[m1 * M2 + m2][i] - x[p][i], 2.0) for i in range(I)])
                if d < d0:
                    d0 = d
                    m10, m20 = m1, m2
        r = q * (r2 - r1) / n_update + r1
        nc = q * (Nc2 - Nc1) / n_update + Nc1
        for m1 in range(M1):
            x1 = m1
            for m2 in range(M2):
                x2 = m2
                if m1 % 2 == 0:
                    x2 += 0.5
                d = sqrt(pow(x1 - m10, 2.0) + pow(x2 - m20, 2.0))
                if int(d) <= nc:
                    for i in range(I):
                        w[m1 * M2 + m2][i] += r * (x[p][i] - w[m1 * M2 + m2][i])
                            
    return w

# def SOFM(n_update, r1, r2, Nc1, Nc2, w, x, I, P): # Use broadcast
#     m_indices = np.array(list(product(range(M1), range(M2))))  # m_indices stores (m1, m2) pairs
#     for q in range(n_update):
#         p = random.randint(0, P - 1)
#         d0 = float('inf')

#         # Calculate distance for all (m1, m2) pairs in one go
#         dists = np.sum((w - x[p]) ** 2, axis=1)
#         m10, m20 = m_indices[np.argmin(dists)]

#         r = q * (r2 - r1) / n_update + r1
#         nc = q * (Nc2 - Nc1) / n_update + Nc1

#         for m1 in range(M1):
#             x1 = m1
#             for m2 in range(M2):
#                 x2 = m2
#                 if m1 % 2 == 0:
#                     x2 += 0.5
#                 d = sqrt(pow(x1 - m10, 2.0) + pow(x2 - m20, 2.0))
#                 if int(d) <= nc:
#                     w[m1 * M2 + m2] += r * (x[p] - w[m1 * M2 + m2])

#     return w



def Calibration(w, x, label0, I, P):
    label = ['.'] * N
    for p in range(P):
        d = np.linalg.norm(w - x[p], axis=1)  # calculate the Euclidean distance
        n0 = np.argmin(d)  # find the index of the smallest distance
        label[n0] = str(label0[p])
    return label



def PrintResult(label):
    for m1 in range(M1):
        if m1 % 2 == 0:
            print(" ", end='')
        for m2 in range(M2):
            print(label[m1 * M2 + m2], end=' ')
        print("\n")
    print("\n")


def PrintResult_figure(label, label_dict, step, color_dict):
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    # ax.axis('off')

    x = []
    y = []
    for m1 in range(M1):
        if m1 % 2 == 0:
            x_offset = 0.5
        else:
            x_offset = 0.0
        for m2 in range(M2):
            x.append(m2 + x_offset)
            y.append(-m1)

    for i, text in enumerate(label):
        fontsize = 8 if text != "." else 12

        # color_dict = {'Iris-setosa': 'black', 'Iris-versicolor': 'black', 'Iris-virginica': 'black'}
        color = color_dict[label_dict[int(text)]] if text != "." else 'black'
        
        if text != ".":
            ax.annotate(text, (x[i], y[i]), ha='center', va='center', color=color, fontsize=fontsize, weight='bold')
        else:
            ax.annotate(text, (x[i], y[i]+0.15), ha='center', va='center', color=color, fontsize=fontsize)
        
        
    # Calculate the count of each class
    label_counts = {}
    for value in label_dict.values():
        label_counts[value] = label_counts.get(value, 0) + 1

    # Generate the legend elements with class counts
    legend_elements = [
        matplotlib.patches.Patch(facecolor=color_dict[key], label=f"{key}: {label_counts[key]}") for key in color_dict]

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), title="Classes")


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim([-1, M2])
    ax.set_ylim([-M1, 1])
    plt.xticks(range(-1, M2+1)) 
    plt.yticks(range(-M1, 1))
    # plt.grid(True)
    plt.title(f"SOFM-{step}th iteration")
    plt.show()
