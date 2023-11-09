import unittest
import numpy as np

from LinearRegression import LinearRegression


class TestLinearRegressionRandomData(unittest.TestCase):
    def test_weights_accuracy(self):
        x = np.random.randn(100, 5) * 100
        w_true = np.random.randn(5) * 5
        b_true = np.random.randn() * 2
        y = np.matmul(x, w_true) + b_true

        leaner = LinearRegression(x, y)
        leaner.fit()
        w_test = leaner.get_unscaled_weights()

        self.assertAlmostEqual(b_true, w_test[0], msg='Incorrect intercept')
        for i in range(len(w_true)):
            self.assertAlmostEqual(w_true[i], w_test[i+1], msg='Incorrect weights')


if __name__ == '__main__':
    unittest.main()

