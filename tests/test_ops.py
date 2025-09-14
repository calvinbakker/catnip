import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from catnip import *

class TestVerified(unittest.TestCase):

    def setUp(self):
        self.sites = 6

    def test_conversion_checks(self):
        """Test conversion between vectors, MPS, matrices, and MPO."""
        # vector-to-MPS-to-vector
        vector = random_vector(self.sites)
        reconstructed_vector = mps_to_vector(vector_to_mps(vector))
        self.assertTrue(np.allclose(vector, reconstructed_vector))

        # MPS-to-vector-to-MPS-to-vector
        mps = random_mps(self.sites, bond_limits=[hsd + 1, 10])
        reconstructed_mps = vector_to_mps(mps_to_vector(mps))
        self.assertTrue(np.allclose(mps_to_vector(mps), mps_to_vector(reconstructed_mps)))

        # matrix-to-MPO-to-matrix
        matrix = random_matrix(self.sites)
        reconstructed_matrix = mpo_to_matrix(matrix_to_mpo(matrix))
        self.assertTrue(np.allclose(matrix, reconstructed_matrix))

        # MPO-to-matrix-to-MPO-to-matrix
        mpo = random_mpo(self.sites, bond_limits=[hsd + 1, 10])
        reconstructed_mpo = matrix_to_mpo(mpo_to_matrix(mpo))
        self.assertTrue(np.allclose(mpo_to_matrix(mpo), mpo_to_matrix(reconstructed_mpo)))

    def test_product_contraction_checks(self):
        """Test product and contraction operations."""
        # vector-matrix product
        vector = random_vector(self.sites)
        matrix = random_matrix(self.sites)
        result1 = matrix @ vector
        result2 = mps_to_vector(mpo_mps_contraction(matrix_to_mpo(matrix), vector_to_mps(vector)))
        self.assertTrue(np.allclose(result1, result2))

        # mps-mpo contraction
        mps = random_mps(self.sites, bond_limits=[hsd + 1, 10])
        mpo = random_mpo(self.sites, bond_limits=[hsd + 1, 10])
        result1 = mps_to_vector(mpo_mps_contraction(mpo, mps))
        result2 = mpo_to_matrix(mpo) @ mps_to_vector(mps)
        self.assertTrue(np.allclose(result1, result2))

        # matrix-matrix product
        matrixA = random_matrix(self.sites)
        matrixB = random_matrix(self.sites)
        result1 = matrixA @ matrixB
        result2 = mpo_to_matrix(mpo_mpo_contraction(matrix_to_mpo(matrixA), matrix_to_mpo(matrixB)))
        self.assertTrue(np.allclose(result1, result2))

        # mpo-mpo contraction
        mpoA = random_mpo(self.sites, bond_limits=[hsd + 1, 10])
        mpoB = random_mpo(self.sites, bond_limits=[hsd + 1, 10])
        result1 = mpo_to_matrix(mpoA) @ mpo_to_matrix(mpoB)
        result2 = mpo_to_matrix(mpo_mpo_contraction(mpoA, mpoB))
        self.assertTrue(np.allclose(result1, result2))

    def test_optimization_checks(self):
        """Test optimization functions."""
        # MPS optimizations
        vector = random_vector(self.sites)
        mps = vector_to_mps(vector)
        self.assertTrue(np.allclose(vector, mps_to_vector(optimize_mps_R(mps))))
        self.assertTrue(np.allclose(vector, mps_to_vector(optimize_mps_L(mps))))
        optimized_mps, entanglement_entropy = mps_entanglement_entropy(mps)
        self.assertTrue(np.allclose(vector, mps_to_vector(optimized_mps)))

        # MPO optimizations
        matrix = random_matrix(self.sites)
        mpo = matrix_to_mpo(matrix)
        self.assertTrue(np.allclose(matrix, mpo_to_matrix(optimize_mpo_L(mpo))))
        self.assertTrue(np.allclose(matrix, mpo_to_matrix(optimize_mpo_R(mpo))))
        optimized_mpo, entanglement_entropy = mpo_entanglement_entropy(mpo)
        self.assertTrue(np.allclose(matrix, mpo_to_matrix(optimized_mpo)))

    def test_final_check(self):
        """Test the final comprehensive check."""
        mps = random_mps(self.sites, bond_limits=[hsd + 1, 10])
        mpo1 = random_mpo(self.sites, bond_limits=[hsd + 1, 10])
        mpo2 = random_mpo(self.sites, bond_limits=[hsd + 1, 10])

        result1_mps, entanglement_entropy1 = mps_entanglement_entropy(
            mpo_mps_contraction(
                mpo_mpo_contraction(
                    mpo1, mpo2
                ),
                mps
            )
        )
        result1 = mps_to_vector(result1_mps)

        result2_mps, entanglement_entropy2 = mps_entanglement_entropy(
            optimize_mps_R(mpo_mps_contraction(
                mpo_mpo_contraction(
                    optimize_mpo_L(mpo1), optimize_mpo_R(mpo2)
                ),
                optimize_mps_L(mps)
            ))
        )
        result2 = mps_to_vector(result2_mps)

        self.assertTrue(np.allclose(result1, result2))
        self.assertTrue(np.allclose(entanglement_entropy1, entanglement_entropy2))

if __name__ == '__main__':
    unittest.main()