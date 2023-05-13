import json
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

# Load patterns and noisy patterns

noisy_level = 15 # 0, 10, 15, 20, 25, 40, 50

noisy_file = "./pattern_"+str(noisy_level)+".json"

with open("./pattern.json", "r") as f:
    original_patterns = json.load(f)

with open(noisy_file, "r") as f:
    noisy_patterns = json.load(f)

n_neuron = 120
n_pattern = 4

# Convert patterns to numpy arrays
for key in original_patterns:
    original_patterns[key] = np.array(original_patterns[key])

for key in noisy_patterns:
    noisy_patterns[key] = np.array(noisy_patterns[key])

# Initialize weight matrix
w = np.zeros((n_neuron, n_neuron))


# Store patterns in the weight matrix
# for key in original_patterns:
#     pattern = original_patterns[key]
#     w += np.outer(pattern, pattern)
# w /= n_pattern
# w = np.floor(w)
# np.fill_diagonal(w, 0)
# print(w)


for key in original_patterns:
    pattern = original_patterns[key]
    print(key,pattern)
    for i in range(n_neuron):
        for j in range(n_neuron):           
            w[i, j] += pattern[i] * pattern[j] 
        # w[i, i] = 0

w = np.floor(w/n_pattern)
np.fill_diagonal(w, 0)
print(w)


# Recall patterns and store in pattern_store
pattern_store = {}

synchronous_update = 1
asynchronous_update = 2
recall_method = 2

for key in noisy_patterns:
    noisy_pattern = noisy_patterns[key]
    original_pattern = original_patterns[key]
    # print(original_pattern)
    # print(noisy_pattern)
    recalled_patterns = [original_pattern.tolist(), noisy_pattern.tolist()]

    while True:
        
        if recall_method == synchronous_update:
            new_pattern = np.where(w @ noisy_pattern >= 0, 1, -1)
            
            if np.array_equal(new_pattern, noisy_pattern):
                break
            
            noisy_pattern = new_pattern
        
        if recall_method == asynchronous_update:

            prev_pattern = noisy_pattern.copy()

            for i in range(n_neuron):
                net_input = w[i] @ noisy_pattern

                noisy_pattern[i] = 1 if net_input >= 0 else -1

            if np.array_equal(noisy_pattern, prev_pattern):
                break
    
        recalled_patterns.append(noisy_pattern.tolist())

    pattern_store[key] = recalled_patterns


def num_to_color(num):
    if num == 1:
        return "white"
    elif num == -1:
        return "red"


max_length = 0
for key, values in pattern_store.items():
    max_length = max(max_length, len(values))

fig, axs = plt.subplots(4, max_length, figsize=(20, 10),gridspec_kw={'wspace': 0.5, 'hspace': 0.5})
spacing = 0.2
row = 0
for key, values in pattern_store.items():
   
    count = 0
    for p in values:
        for i in range(10):
            for j in range(12):
                color = num_to_color(p[i * 12 + j])
                axs[row, count].scatter(j * spacing, -i * spacing, c=color, s=50)
        axs[row, count].axis("off")
        axs[row, count].set_xlim(-0.5 * spacing, 11.5 * spacing)
        axs[row, count].set_ylim(-9.5 * spacing, 0.5 * spacing)
        if count == 0:
            axs[row, count].set_title(f"{key}-original")
        if count == 1:
            axs[row, count].set_title(f"noisy_level:{noisy_level}%")
        if count > 1:
            axs[row, count].set_title(f"{count-1}th-iter")
        count += 1
    row += 1

for i in range(row):
    for j in range(max_length):
         axs[i, j].axis("off")

plt.show()
        
