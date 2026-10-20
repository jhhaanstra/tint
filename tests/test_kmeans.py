import numpy as np
from numpy.testing import assert_array_equal

from src.tint.kmeans import (
    _closest_centroid,
    _compute_means,
    _create_clusters,
    _euclidean_distance,
)
from tint.kmeans import _initialize_random_centroids


def test_initialize_random_centroids():
    np.random.seed(42)
    centroids = _initialize_random_centroids(
        3, np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    )

    expected = np.array([[5, 6], [7, 8], [1, 2]])

    assert_array_equal(centroids, expected)


def test_euclidian_distance():
    assert _euclidean_distance(np.array([6, 7]), np.array([4, 2])) == np.float64(
        5.385164807134504
    )
    assert _euclidean_distance(np.array([6, -7]), np.array([-4, 2])) == np.float64(
        13.45362404707371
    )

def test_closest_centroid():
    c1 = np.array([0, 0])
    c2 = np.array([11, 11])

    centroids = np.array([c1, c2])
    assert _closest_centroid(np.array([1, 2]), centroids, 2) == 0
    assert _closest_centroid(np.array([3, 4]), centroids, 2) == 0
    # This one is perfectly tied, but returns the first of the tying centroids
    assert _closest_centroid(np.array([5, 6]), centroids, 2) == 0
    assert _closest_centroid(np.array([7, 8]), centroids, 2) == 1
    assert _closest_centroid(np.array([9, 10]), centroids, 2) == 1



def test_create_clusters():
    centroids = np.array([
        [0, 0],
        [10, 10]
    ])

    data = np.array([
        [1, 1],
        [2, 3],
        [8, 9],
        [10, 11],
        [3, 2]
    ])

    expected = np.array([0, 0, 1, 1, 0])
    result = _create_clusters(centroids, 2, data)
    assert_array_equal(result, expected)



def test_compute_means():
    data = np.array([
        [1, 2],
        [3, 4],
        [10, 12],
        [12, 14],
        [5, 6]
    ])

    expected = np.array(
        [
            # Mean of [1,2], [3,4], [5,6]
            [3, 4],
            # Mean of [10,12], [12,14] -> [((10 + 12) / 2), ((12 + 14) / 2)]
            [11, 13],
        ]
    )

    result = _compute_means(np.array([0, 0, 1, 1, 0]), 2, data, np.array(2))
    assert_array_equal(result, expected)