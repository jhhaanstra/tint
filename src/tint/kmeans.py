import numpy as np
from matplotlib import pyplot as plt
from numpy.core.multiarray import ndarray
from sklearn.datasets import make_blobs


def _initialize_random_centroids(K, data) -> ndarray:
    """Initializes and returns k random centroids"""
    m, n = np.shape(data)
    # a centroid should be of shape (1, n), so the centroids array will be of shape (K, n)
    centroids = np.empty((K, n))
    for i in range(K):
        # pick a random data point from X as the centroid
        centroids[i] = data[np.random.choice(range(m))]
    return centroids


def _euclidean_distance(vec1, vec2):
    """Calculates and returns the Euclidean distance between two vectors vec1 and vec2"""
    return np.sqrt(np.sum(np.power(vec1 - vec2, 2)))


def _closest_centroid(vector, centroids, K):
    """Finds and returns the index of the closest centroid for a given vector"""
    distances = np.empty(K)
    for i in range(K):
        distances[i] = _euclidean_distance(centroids[i], vector)
    return np.argmin(distances)


def _create_clusters(centroids, K, data):
    """Returns an array of cluster indices for all the data samples"""
    m, _ = np.shape(data)
    cluster_idx = np.empty(m)
    for i in range(m):
        cluster_idx[i] = _closest_centroid(data[i], centroids, K)
    return cluster_idx


def _compute_means(cluster_idx, K, X, previous_centroids):
    """Computes and returns the new centroids of the clusters"""
    _, n = np.shape(X)
    centroids = np.empty((K, n))
    for i in range(K):
        points = X[cluster_idx == i]  # gather points for the cluster i
        if len(points) > 0:
            centroids[i] = np.mean(points, axis=0)
        else:
            centroids[i] = previous_centroids[i]

    return centroids


def kmeans(K, X, max_iterations=500):
    """Runs the K-means algorithm and computes the final clusters"""
    centroids = _initialize_random_centroids(K, X)

    clusters = _create_clusters(centroids, K, X)
    for _ in range(max_iterations):
        previous_centroids = centroids
        clusters = _create_clusters(centroids, K, X)
        centroids = _compute_means(clusters, K, X, previous_centroids)

        if not (previous_centroids - centroids).any():
            break

    return clusters, centroids


if __name__ == "__main__":
    np.random.seed(23)
    X, y = make_blobs(n_samples=500, n_features=3, centers=3, random_state=123)

    # Plot each point, colored by its cluster
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        c=kmeans(3, X),
    )

    plt.title("K-means Clustering")
    plt.show()
