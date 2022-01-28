
import numpy as np

from distances.euclidean import euclidean_distance
from persistence import get_indices_of_birth_death_persistence_diagrams, generate_distance_matrix, \
    get_persistence_diagrams_from_indices


def generate_point_cloud(number_of_points: int, points_dimension: int, amplifier: int = 10):
    assert number_of_points > 0 and points_dimension >= 1
    return amplifier * np.random.rand(number_of_points, points_dimension)


if __name__ == "__main__":
    hom_dim = 0
    number_of_points = 3
    number_of_dimensions = 5
    point_cloud = generate_point_cloud(number_of_points, number_of_dimensions)

    tril_distance_matrix, sparse_distance_matrix = generate_distance_matrix(point_cloud, euclidean_distance)
    indices_for_persistence_pairs = get_indices_of_birth_death_persistence_diagrams(tril_distance_matrix,
                                                                                    sparse_distance_matrix,
                                                                                    hom_dim)
    dgm_from_indices = get_persistence_diagrams_from_indices(sparse_distance_matrix,
                                                             indices_for_persistence_pairs)

