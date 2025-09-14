import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from catnip import *

class TestEvolution(unittest.TestCase):
    def setUp(self):
        self.sites = 25
        self.timesteps = 2
        self.amount_of_tests = 10
        self.amount_of_decimals_in_rounding = 3
        self.random_seed = 42 
        np.random.seed(self.random_seed)

    def test_multiple_rules_no_noise(self):
        """Test consistency for multiple random rule numbers and initial conditions with no noise."""
        for test_idx in range(self.amount_of_tests):
            rule_number = np.random.randint(0, 256)
            initial_condition = np.random.randint(0, hsd, size=self.sites)
            with self.subTest(test=test_idx, rule=rule_number, initial=initial_condition[:5]):  # Show first 5 elements for readability
                result_bits = evolution_with_bits(rule_number, initial_condition, self.timesteps)
                result_noisy = evolution_with_tensors(rule_number, initial_condition, self.timesteps, noise_level=0)
                # They should be the same when noise_level=0
                result_bits_rounded = np.round(result_bits, self.amount_of_decimals_in_rounding)
                result_noisy_rounded = np.round(result_noisy, self.amount_of_decimals_in_rounding)
                np.testing.assert_array_equal(result_bits_rounded, result_noisy_rounded)

    def test_with_noise_differs(self):
        """Test that with noise, the results differ for multiple random configurations."""
        for test_idx in range(self.amount_of_tests):
            rule_number = np.random.randint(0, 256)
            initial_condition = np.random.randint(0, hsd, size=self.sites)
            with self.subTest(test=test_idx, rule=rule_number, initial=initial_condition[:5]):
                result_bits = evolution_with_bits(rule_number, initial_condition, self.timesteps)
                result_noisy = evolution_with_tensors(rule_number, initial_condition, self.timesteps, noise_level=1e-3)
                # They should differ due to noise
                self.assertFalse(np.array_equal(result_bits, result_noisy))


if __name__ == '__main__':
    unittest.main()