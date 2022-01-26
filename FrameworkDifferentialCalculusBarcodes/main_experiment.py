import gudhi 
import numpy as np

def generate_point_cloud(number_of_points: int, points_dimension: int, amplifier: int=10):
    assert number_of_points > 0 and points_dimension >= 1
    return amplifier*np.random.rand(number_of_points, points_dimension)

def generate_rips_complex(point_cloud: np.array, max_dimension: int):
    assert max_dimension >= 0
    points = point_cloud.tolist()
    rips_complex = gudhi.RipsComplex(points=points)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    return simplex_tree

def compute_persistence(rips_complex_simplex_tree, hom_coeff: int = 2):
    persistence_diagrams = rips_complex_simplex_tree.persistence(homology_coeff_field=hom_coeff)
    persistence_pairs = rips_complex_simplex_tree.persistence_pairs()
    return persistence_diagrams, persistence_pairs

if __name__ == "__main__":
    point_cloud = generate_point_cloud(50, 5)
    rips_complex = generate_rips_complex(point_cloud)
    persistence_diagrams, persistence_pairs = compute_persistence(rips_complex)
    print(persistence_diagrams)
    print("======")
    print(persistence_pairs)