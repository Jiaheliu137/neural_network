import json
import matplotlib.pyplot as plt


with open("./pattern.json", "r") as f:
    data = json.load(f)

def num_to_color(num):
    if num == 1:
        return "white"
    elif num == -1:
        return "red"

def draw_patterns(data, spacing=1):

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for index, (pattern_key, pattern_value) in enumerate(data.items()):
        print(pattern_value)
        row_idx, col_idx = divmod(index, 2)
        for i in range(10):
            for j in range(12):
                color = num_to_color(pattern_value[i * 12 + j])
                axs[row_idx, col_idx].scatter(j * spacing, -i * spacing, c=color, s=200)

        axs[row_idx, col_idx].axis("off") 
        axs[row_idx, col_idx].set_xlim(-0.5 * spacing, 11.5 * spacing)
        axs[row_idx, col_idx].set_ylim(-9.5 * spacing, 0.5 * spacing)
        axs[row_idx, col_idx].set_title(pattern_key)

    plt.show()

draw_patterns(data, spacing=1)  

