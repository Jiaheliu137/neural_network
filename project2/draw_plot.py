import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *

n_parity = 4
inputs, outputs = generate_parity_dataset(4)

learning_rates = [0.5, 1.0, 1.5, 2, 2.5, 3.0]
hidden_neurons_list = [4, 6, 8, 10, 12, 14]
num_epochs = 5000

num_repeats = 10


fig, axes = plt.subplots(len(learning_rates), len(hidden_neurons_list), figsize=(15, 10), sharex=True, sharey=True) 
fig.tight_layout(pad=4.0)

total_combinations = len(learning_rates) * len(hidden_neurons_list) * num_repeats
progress_bar = tqdm(total=total_combinations, desc="Training progress")

for i, lr in enumerate(learning_rates):
    for j, hidden_neurons in enumerate(hidden_neurons_list):
        accuracies = []
        max_loss = -np.inf
        min_loss = np.inf
        for repeat in range(num_repeats):
            input_size = inputs.shape[1]
            output_size = outputs.shape[1]
            weights_input_hidden = np.random.uniform(-1, 1, size=(input_size, hidden_neurons))
            weights_hidden_output = np.random.uniform(-1, 1, size=(hidden_neurons, output_size))

            weights_input_hidden, weights_hidden_output, loss_list = train(inputs, outputs, weights_input_hidden, weights_hidden_output, lr, num_epochs)

            axes[i, j].plot(range(num_epochs), loss_list, alpha=0.3)

            max_loss = max(max_loss, np.max(loss_list))
            min_loss = min(min_loss, np.min(loss_list))

            predictions = test(inputs, weights_input_hidden, weights_hidden_output)
            accuracy = np.mean(np.round(predictions) == outputs)
            accuracies.append(accuracy)

            progress_bar.update(1)

        mean_accuracy = np.mean(accuracies)
        axes[i, j].set_title(f"LR:{lr}, Hidden:{hidden_neurons}, Acc:{mean_accuracy:.2f}")
        axes[i, j].set_ylim(min_loss, max_loss)
        axes[i, j].set_xlabel("Epoch")
        axes[i, j].set_ylabel("Loss")

progress_bar.close()
plt.show()

