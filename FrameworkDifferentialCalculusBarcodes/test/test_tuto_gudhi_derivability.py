import numpy as np
import tensorflow as tf

from auxiliar.tuto_code import RipsModel
from differentiable_homology.ppdd_continuation import generate_differentiable_persistence_diagrams


def generate_point_cloud(number_of_points: int, points_dimension: int, amplifier: int = 10):
    assert number_of_points > 0 and points_dimension >= 1
    return amplifier * np.random.rand(number_of_points, points_dimension)


def call_our_own(point_cloud, hom_dim, number_of_points_in_dgm):
    point_cloud_init = tf.identity(point_cloud)
    point_cloud_var = tf.Variable(initial_value=point_cloud_init, trainable=True)
    with tf.GradientTape() as tape:
        dgm_from_indices = generate_differentiable_persistence_diagrams(point_cloud_var, hom_dim, number_of_points_in_dgm)
        persistences = dgm_from_indices[:, 1] - dgm_from_indices[:, 0]
        loss = tf.math.reduce_sum(persistences)
    gradient = tape.gradient(loss, point_cloud_var)
    return gradient


def call_tuto(point_cloud, card, hom):
    ml = 10000  # max distance in Rips

    Xinit = tf.identity(point_cloud)
    X = tf.Variable(initial_value=Xinit, trainable=True)

    model = RipsModel(X=X, mel=ml, dim=hom, card=card)

    with tf.GradientTape() as tape:
        # Compute persistence diagram
        dgm = model.call()
        # Loss
        persistences = dgm[:, 1] - dgm[:, 0]
        loss = tf.math.reduce_sum(persistences)
    # Compute and apply gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    return gradients


def test_same_gradients_for_our_own_and_diff_tuto():
    hom_dim = 1
    number_of_points = 10
    number_of_dimensions = 2
    number_of_points_in_dgm = 5
    # Test several times with different values of the point cloud
    for i in range(10):
        point_cloud = generate_point_cloud(number_of_points, number_of_dimensions)
        gradients_tuto = call_tuto(point_cloud, number_of_points_in_dgm, hom_dim)
        gradients_own = call_our_own(point_cloud, hom_dim, number_of_points_in_dgm)
        assert np.allclose(gradients_tuto[0].numpy(), gradients_own.numpy())
