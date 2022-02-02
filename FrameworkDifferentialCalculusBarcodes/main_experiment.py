import os

import imageio as imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from differentiable_homology.ppdd_continuation import generate_differentiable_persistence_diagrams


def create_path(dir_path):
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass


def build_gif(gif_path, frames_per_image=2, extension='png'):
    # Build GIF
    filenames = list(map(lambda filename: int(filename[:-4]), os.listdir(gif_path)))
    filenames.sort()
    with imageio.get_writer(f'{gif_path}/result.gif', mode='I') as writer:
        for filename in [f'{gif_path}/{gif_filename}.{extension}' for gif_filename in filenames]:
            image = imageio.imread(filename)
            for _ in range(frames_per_image):
                writer.append_data(image)


def paint_point_cloud(point_cloud, gif_path, number, x_low, x_high, y_low, y_high):
    plt.clf()
    if number == -1:
        plt.title(f'Initial position')
    else:
        plt.title(f'Epoch {number}')
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1])
    plt.ylim(y_low, y_high)
    plt.xlim(x_low, x_high)
    plt.plot()
    plt.savefig(f'{gif_path}/{number}.png')


def generate_point_cloud(number_of_points: int, points_dimension: int, amplifier: int = 10):
    assert number_of_points > 0 and points_dimension >= 1
    return amplifier * np.random.rand(number_of_points, points_dimension)


def gradient_descent(point_cloud, hom_dim, number_of_points_in_dgm, n_epochs, gif_path, x_low, x_high, y_low, y_high):
    paint_point_cloud(point_cloud, gif_path, -1, x_low, x_high, y_low, y_high)
    point_cloud_init = tf.identity(point_cloud)
    point_cloud_var = tf.Variable(initial_value=point_cloud_init, trainable=True)
    optimizer = tf.keras.optimizers.Adam()
    for epoch in range(n_epochs + 1):
        with tf.GradientTape() as tape:
            dgm_from_indices = generate_differentiable_persistence_diagrams(point_cloud_var, hom_dim,
                                                                            number_of_points_in_dgm)
            # persistences = dgm_from_indices[:, 1] - dgm_from_indices[:, 0]
            # loss = tf.math.reduce_sum(persistences)
            loss = -tf.math.reduce_sum(tf.square(.5 * (dgm_from_indices[:, 1] - dgm_from_indices[:, 0])))
        gradients = tape.gradient(loss, point_cloud_var)
        optimizer.apply_gradients(zip([gradients], [point_cloud_var]))
        paint_point_cloud(point_cloud_var, gif_path, epoch, x_low, x_high, y_low, y_high)


def main():
    gif_folder_path = './gif'
    create_path(gif_folder_path)
    hom_dim = 1
    number_of_points = 300
    number_of_dimensions = 2
    n_epochs = 100
    number_of_points_in_dgm = 50
    amplifier = 1
    init_point_cloud = generate_point_cloud(number_of_points, number_of_dimensions, amplifier)
    x_low, x_high = -0.1, amplifier + 0.1
    y_low, y_high = -0.1, amplifier + 0.1
    gradient_descent(init_point_cloud, hom_dim, number_of_points_in_dgm, n_epochs, gif_folder_path,
                     x_low, x_high, y_low, y_high)
    build_gif(gif_folder_path)


if __name__ == "__main__":
    main()
