from matplotlib import pyplot as plt

from src.tint.kmeans import kmeans
from pathlib import Path

import numpy as np
from PIL import Image

if __name__ == "__main__":
    image = Image.open(Path(__file__).parent / "images/1.jpg").convert("RGB")
    image.thumbnail((100, 100))

    # flatten
    image_data = np.array(image).reshape(-1, 3)
    colors, centroids = kmeans(4, image_data)

    classified_colors = centroids[colors.astype(int)] / 255.0

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(
        image_data[:, 0],
        image_data[:, 1],
        image_data[:, 2],
        c=classified_colors
    )

    plt.title("Image tints")
    plt.show()

