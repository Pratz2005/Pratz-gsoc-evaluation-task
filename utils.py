import numpy as np
import matplotlib.pyplot as plt

def load_data():
    images = np.load("data/images.npy")
    labels = np.load("data/labels.npy")
    return images, labels

def plot_pixel_intensity_distribution(images, labels, title="Pixel Intensity Distribution"):
    # Flatten pixel values for both classes
    pixel_values_0 = images[labels == 0].flatten()
    pixel_values_1 = images[labels == 1].flatten()

    # Create histogram plot
    plt.figure(figsize=(8, 4))
    plt.hist(pixel_values_0, bins=50, alpha=0.5, label="Label 0")
    plt.hist(pixel_values_1, bins=50, alpha=0.5, label="Label 1")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.show()
