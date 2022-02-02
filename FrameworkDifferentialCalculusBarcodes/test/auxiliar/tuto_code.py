import numpy as np
import tensorflow as tf
import gudhi as gd

from distances.euclidean import euclidean_distance_tensorflow


def Rips(DX, mel, dim, card):
    # Parameters: DX (distance matrix),
    #             mel (maximum edge length for Rips filtration),
    #             dim (homological dimension),
    #             card (number of persistence diagram points, sorted by distance-to-diagonal)

    # Compute the persistence pairs with Gudhi
    rc = gd.RipsComplex(distance_matrix=DX, max_edge_length=mel)
    st = rc.create_simplex_tree(max_dimension=dim + 1)
    dgm = st.persistence()
    pairs = st.persistence_pairs()

    # Retrieve vertices v_a and v_b by picking the ones achieving the maximal
    # distance among all pairwise distances between the simplex vertices
    indices, pers = [], []
    for s1, s2 in pairs:
        if len(s1) == dim + 1:
            l1, l2 = np.array(s1), np.array(s2)
            i1 = [s1[v] for v in np.unravel_index(np.argmax(DX[l1, :][:, l1]), [len(s1), len(s1)])]
            i2 = [s2[v] for v in np.unravel_index(np.argmax(DX[l2, :][:, l2]), [len(s2), len(s2)])]
            indices += i1
            indices += i2
            pers.append(st.filtration(s2) - st.filtration(s1))

    # Sort points with distance-to-diagonal
    perm = np.argsort(pers)
    indices = list(np.reshape(indices, [-1, 4])[perm][::-1, :].flatten())
    # Output indices
    indices = indices[:4 * card] + [0 for _ in range(0, max(0, 4 * card - len(indices)))]
    return list(np.array(indices, dtype=np.int32))


class RipsModel(tf.keras.Model):
    def __init__(self, X, mel=12, dim=1, card=50):
        super(RipsModel, self).__init__()
        self.X = X
        self.mel = mel
        self.dim = dim
        self.card = card

    def call(self):
        m, d, c = self.mel, self.dim, self.card

        # Compute distance matrix
        DX = euclidean_distance_tensorflow(self.X, self.X.shape[0], self.X.shape[1])
        DXX = tf.reshape(DX, [1, DX.shape[0], DX.shape[1]])

        # Turn numpy function into tensorflow function
        RipsTF = lambda DX: tf.numpy_function(Rips, [DX, m, d, c], [tf.int32 for _ in range(4 * c)])

        # Compute vertices associated to positive and negative simplices
        # Don't compute gradient for this operation
        ids = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(RipsTF, DXX, dtype=[tf.int32 for _ in range(4 * c)]))
        # Get persistence diagram by simply picking the corresponding entries in the distance matrix
        dgm = tf.reshape(tf.gather_nd(DX, tf.reshape(ids, [2 * c, 2])), [c, 2])
        return dgm
