from typing import Callable, List
import tensorflow as tf

import gudhi
import numpy as np

from distances.euclidean import euclidean_distance


def generate_distance_matrix(point_cloud: np.array,
                             distance_function: Callable[[np.array, np.array], float] = euclidean_distance):
    numpy_distance_matrix = np.zeros((point_cloud.shape[0], point_cloud.shape[0]))
    tril_indices = zip(*np.tril_indices(point_cloud.shape[0]))
    for x, y in tril_indices:
        if x != y:
            pairwise_distance = distance_function(point_cloud[x, :], point_cloud[y, :])
            numpy_distance_matrix[x, y] = pairwise_distance
    return numpy_distance_matrix


def generate_tril_distance_matrix_for_gudhi(distance_matrix):
    tril_distance_matrix = []
    tril_indices = zip(*np.tril_indices(distance_matrix.shape[0]))
    current_row = []
    for x, y in tril_indices:
        if x == y:
            tril_distance_matrix.append(current_row[:])
            current_row = []
        else:
            pairwise_distance = distance_matrix[x, y]
            current_row.append(pairwise_distance)
    return tril_distance_matrix


def generate_rips_complex(distance_matrix: List[List[float]], max_dimension: int):
    assert max_dimension >= 0
    rips_complex = gudhi.RipsComplex(distance_matrix=distance_matrix)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension + 1)
    return simplex_tree


def _create_dict_of_dims_for_pdgm_points(persistence_diagrams):
    points_dict = dict()
    for pdgm_point in persistence_diagrams:
        if pdgm_point[1] not in points_dict:
            points_dict[pdgm_point[1]] = [pdgm_point[0]]
        else:
            points_dict[pdgm_point[1]].append(pdgm_point[0])
    return points_dict


def _associate_dimension_to_persistence_pairs(rips_complex_simplex_tree, persistence_diagrams, persistence_pairs):
    map_point_per_dgm_to_dim = _create_dict_of_dims_for_pdgm_points(persistence_diagrams)
    persistence_pairs_with_dim = []
    for pp in persistence_pairs:
        birth, death = rips_complex_simplex_tree.filtration(pp[0]), rips_complex_simplex_tree.filtration(pp[1])
        dims = map_point_per_dgm_to_dim[(birth, death)]
        persistence_pairs_with_dim.append((pp, dims))
    return persistence_pairs_with_dim


def compute_persistence(rips_complex_simplex_tree, hom_coeff: int = 2):
    persistence_diagrams = rips_complex_simplex_tree.persistence(homology_coeff_field=hom_coeff + 1)
    persistence_pairs = rips_complex_simplex_tree.persistence_pairs()
    persistence_pairs_with_dim = _associate_dimension_to_persistence_pairs(rips_complex_simplex_tree,
                                                                           persistence_diagrams, persistence_pairs)
    return persistence_diagrams, persistence_pairs_with_dim


def extract_persistence_pairs_with_given_dimension(persistence_pairs, hom_dim: int):
    return list(map(lambda pp: pp[0], filter(lambda pp: hom_dim in pp[1], persistence_pairs)))


# Here, we compute v_bar and w_bar for each simplex of each persistence pair, to give the derivative definition
# seen in "A Framework for Differential Calculus on Persistence Barcodes". The ordering is given by the ordering
# yield by Gudhi.
def compute_indices_persistence_pairs(rips_complex, distance_matrix: np.array, persistence_pairs, number_of_points_sampled: int):
    indices = []
    pers = []
    for s1, s2 in persistence_pairs:
        if len(s1) != 0 and len(
                s2) != 0:  # We discard points dying at infinity, specially the max. connected component for H_0 group.
            l1, l2 = np.array(s1), np.array(s2)
            i1 = [s1[v] for v in np.unravel_index(np.argmax(distance_matrix[l1, :][:, l1]), [len(s1), len(s1)])]
            i2 = [s2[v] for v in np.unravel_index(np.argmax(distance_matrix[l2, :][:, l2]), [len(s2), len(s2)])]
            pers.append(rips_complex.filtration(s2) - rips_complex.filtration(s1))
            indices += i1
            indices += i2
    # Sort points with distance-to-diagonal
    perm = np.argsort(pers)
    indices = list(np.reshape(indices, [-1, 4])[perm][::-1, :].flatten())

    # Output indices
    indices = indices[:4 * number_of_points_sampled] + [0 for _ in range(0, max(0, 4 * number_of_points_sampled - len(indices)))]
    return list(np.array(indices, dtype=np.int32))


def get_persistence_diagrams_from_indices(distance_matrix: tf.Tensor, indices: List[List[int]], number_of_points_in_dgm: int):
    dgm = tf.reshape(tf.gather_nd(distance_matrix, tf.reshape(indices, [2 * number_of_points_in_dgm, 2])),
                     [number_of_points_in_dgm, 2])
    return dgm


def get_persistence_diagrams_from_indices_correct(distance_matrix: np.array, indices: List[List[int]]):
    dgm = []
    for birth_idxs, death_idxs in indices:
        birth, death = distance_matrix[birth_idxs[0], birth_idxs[1]], distance_matrix[death_idxs[0], death_idxs[1]]
        dgm.append([birth, death])
    return np.array(dgm)


def get_indices_of_birth_death_persistence_diagrams(distance_matrix,
                                                    hom_dim, number_of_points_sampled,
                                                    hom_coeff: int = 2):
    rips_complex = generate_rips_complex(distance_matrix, hom_dim)
    _, persistence_pairs = compute_persistence(rips_complex, hom_coeff)
    persistence_pairs_int_dimension = extract_persistence_pairs_with_given_dimension(persistence_pairs,
                                                                                     hom_dim)
    indices_for_persistence_pairs = compute_indices_persistence_pairs(rips_complex,
                                                                      distance_matrix,
                                                                      persistence_pairs_int_dimension,
                                                                      number_of_points_sampled)
    return indices_for_persistence_pairs
