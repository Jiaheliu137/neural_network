import matplotlib.pyplot as plt

color_list = ['red', 'blue', 'green', 'purple', 'cyan', 'orange', 'black', 'magenta', 'lime', 'navy', 'darkred', 'darkgreen', 'gold', 'teal']

plt.figure(figsize=(15, 2))

for i, color in enumerate(color_list):
    plt.bar(i, 1, color=color, edgecolor=color, width=1)

plt.xlim(-0.5, len(color_list)-0.5)
plt.ylim(0, 1)
plt.axis('off')

for i, color in enumerate(color_list):
    plt.text(i, 0.5, color, ha='center', va='center', color='white' if color in ['black', 'navy', 'darkred', 'darkgreen'] else 'black')

plt.show()

import numpy as np

# 读取文本文件到NumPy数组
data = np.loadtxt('./clusters_4.txt', dtype=str)

print(data)
