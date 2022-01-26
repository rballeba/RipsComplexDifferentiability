from typing import Callable, List
from scipy.sparse import tril

import gudhi
import numpy as np

from distances.euclidean import euclidean_distance


def generate_point_cloud(number_of_points: int, points_dimension: int, amplifier: int = 10):
    assert number_of_points > 0 and points_dimension >= 1
    return amplifier * np.random.rand(number_of_points, points_dimension)


def generate_distance_matrix(point_cloud: np.array,
                             distance_function: Callable[[np.array, np.array], float] = euclidean_distance):
    numpy_distance_matrix = np.zeros(point_cloud.shape[0], point_cloud.shape[0])
    tril_distance_matrix = []
    tril_indices = zip(*np.tril_indices(point_cloud.shape[0]))
    current_row = []
    for x, y in tril_indices:
        if x == y:
            tril_distance_matrix.append(current_row[:])
            current_row = []
        else:
            pairwise_distance = distance_function(point_cloud[x, :], point_cloud[y, :])
            current_row.append(pairwise_distance)
            numpy_distance_matrix[x, y] = pairwise_distance
    return tril_distance_matrix, tril(numpy_distance_matrix, format='csc')


def generate_rips_complex(distance_matrix: List[List[float]], max_dimension: int):
    assert max_dimension >= 0
    rips_complex = gudhi.RipsComplex(distance_matrix=distance_matrix)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    return simplex_tree


def compute_persistence(rips_complex_simplex_tree, hom_coeff: int = 2):
    persistence_diagrams = rips_complex_simplex_tree.persistence(homology_coeff_field=hom_coeff + 1)
    persistence_pairs = rips_complex_simplex_tree.persistence_pairs()
    return persistence_diagrams, persistence_pairs


# Here, we compute v_bar and w_bar for each simplex of each persistence pair, to give the derivative definition
# seen in "A Framework for Differential Calculus on Persistence Barcodes". The ordering is given by the ordering
# yield by Gudhi.
def compute_indices_persistence_pairs(distance_matrix: np.array, persistence_pairs):
    indices = []
    for s1, s2 in persistence_pairs:
        l1, l2 = np.array(s1), np.array(s2)
        i1 = [s1[v] for v in np.unravel_index(np.argmax(distance_matrix[l1, :][:, l1]), [len(s1), len(s1)])]
        i2 = [s2[v] for v in np.unravel_index(np.argmax(distance_matrix[l2, :][:, l2]), [len(s2), len(s2)])]
        indices += (i1, i2)
    return indices


if __name__ == "__main__":
    max_dimension = 2
    point_cloud = generate_point_cloud(3, 5)
    tril_distance_matrix, sparse_distance_matrix = generate_distance_matrix(point_cloud)
    rips_complex = generate_rips_complex(tril_distance_matrix, max_dimension)
    persistence_diagrams, persistence_pairs = compute_persistence(rips_complex)
    indices_for_persistence_pairs = compute_indices_persistence_pairs(sparse_distance_matrix, persistence_pairs)
    #TODO, differentiate by dimensions when getting persistence indices and pairs
    #TODO get u by differentiating The map Bar -> N (in our case R)
