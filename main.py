import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from tint.kmeans import kmeans


def plot(colors):
    _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(
        image_data[:, 0],
        image_data[:, 1],
        image_data[:, 2],
        c=colors
    )
    plt.title("Image tints")
    plt.show()


def parse_args():
    global args
    parser = argparse.ArgumentParser(
        prog='tint',
        description='Extract a color pallette from a given image')
    parser.add_argument(
        'image_path',
        type=Path,
        help='Path to the image'
    )
    parser.add_argument(
        '--nr-colors',
        type=int,
        default=4,
        help='Number of colors'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=100,
        help='Width of the image to downscale to'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=100,
        help='Height of the image to downscale to'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    image = Image.open(args.image_path)
    image.thumbnail((args.width, args.height))

    # flatten
    image_data = np.array(image).reshape(-1, 3)
    colors, centroids = kmeans(args.nr_colors, image_data)

    classified_colors = centroids[colors.astype(int)] / 255.0

    for centroids in centroids:
        print({'R': int(centroids[0]), 'G': int(centroids[1]), 'B': int(centroids[2])})

    plot(classified_colors)
