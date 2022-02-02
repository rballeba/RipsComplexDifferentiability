import numpy as np
import tensorflow as tf

from distances.euclidean import euclidean_distance_tensorflow
from persistence import get_indices_of_birth_death_persistence_diagrams, get_persistence_diagrams_from_indices


def generate_differentiable_persistence_diagrams(point_cloud: np.array, hom_dim: int, number_of_points_in_dgm: int):
    number_of_points = point_cloud.shape[0]
    number_of_dimensions = point_cloud.shape[1]
    distance_matrix = euclidean_distance_tensorflow(point_cloud, number_of_points, number_of_dimensions)
    DXX = tf.reshape(distance_matrix, [1, distance_matrix.shape[0], distance_matrix.shape[1]])
    # Turn numpy function into tensorflow function
    RipsTF = lambda DX: tf.numpy_function(get_indices_of_birth_death_persistence_diagrams,
                                          [DX, hom_dim, number_of_points_in_dgm],
                                          [tf.int32 for _ in range(4 * number_of_points_in_dgm)])

    # Compute vertices associated to positive and negative simplices
    # Don't compute gradient for this operation
    indices_for_persistence_pairs = tf.nest.map_structure(tf.stop_gradient,
                                                          tf.map_fn(RipsTF, DXX,
                                                                    dtype=[tf.int32 for _ in
                                                                           range(4 * number_of_points_in_dgm)]))
    dgm_from_indices = get_persistence_diagrams_from_indices(distance_matrix, indices_for_persistence_pairs,
                                                             number_of_points_in_dgm)

    return dgm_from_indices
