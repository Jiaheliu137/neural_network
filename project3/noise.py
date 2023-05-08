import json
import numpy as np

def load_patterns(file_path):
    with open(file_path, 'r') as f:
        patterns = json.load(f)
    return {int(key): np.array(value) for key, value in patterns.items()}

def add_noise(patterns, noise_level):
    noisy_patterns = {}
    for key, pattern in patterns.items():
        n_pixels = len(pattern)
        # Generate an array of random values between 0 and 1
        random_values = np.random.rand(n_pixels)
        # Create a mask where values are less than noise_level
        noise_mask = random_values < noise_level
        # Use np.where to flip the sign of the pixels where noise_mask is True
        noisy_pattern = np.where(noise_mask, -pattern, pattern)
        noisy_patterns[key] = noisy_pattern.tolist()
    return noisy_patterns

def save_patterns(patterns, file_path):
    with open(file_path, 'w') as f:
        json.dump(patterns, f)

def main():
    noise_levels = [0.0, 0.1, 0.15, 0.20, 0.25,0.40, 0.50]
    patterns = load_patterns('./pattern.json')

    for noise_level in noise_levels:
        noisy_patterns = add_noise(patterns, noise_level)
        save_path = f'./pattern_{int(noise_level * 100)}.json'
        save_patterns(noisy_patterns, save_path)

if __name__ == '__main__':
    main()
