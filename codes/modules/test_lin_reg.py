import unittest
import numpy as np
import pandas as pd
from modules.lin_reg import LinearRegression  

class TestLinearRegressionPerfect(unittest.TestCase):
    def setUp(self):
        # Create a simple dataset with known solution
        # y = 2x + 1
        self.X = np.array([[1], [2], [3], [4], [5]])
        self.y = np.array([3, 5, 7, 9, 11])
        self.model = LinearRegression(add_bias=True)
        self.model.fit(self.X, self.y)

    def test_coefficients(self):
        # Expect intercept ~1 and slope ~2
        np.testing.assert_allclose(self.model.beta, [1.0, 2.0], rtol=1e-5)

    def test_predictions(self):
        y_pred = self.model.predict(self.X)
        np.testing.assert_allclose(y_pred, self.y, rtol=1e-5)

    def test_residuals_zero(self):
        res = self.model.residuals(self.X, self.y)
        np.testing.assert_allclose(res, np.zeros_like(self.y), atol=1e-8)

    def test_r_squared(self):
        R, R_adj = self.model.R_squared()
        self.assertAlmostEqual(R, 1.0, places=8)
        self.assertAlmostEqual(R_adj, 1.0, places=8)

    def test_residual_standard_error(self):
        rse = self.model.residuals_SE()
        self.assertAlmostEqual(rse, 0.0, places=8)

    def test_f_statistic(self):
        F, p = self.model.F_score()
        self.assertGreater(F, 1e6)     # Should be very large (perfect fit)
        self.assertLess(p, 1e-10)      # p-value should be tiny

class TestLinearRegressionUSJudges(unittest.TestCase):
    def setUp(self):
        # First we load in the dataset (which we borrowed from R datasets library)
        # Testing against output from R
        df = pd.read_csv("data/USJudgeRatings.csv", index_col=0)
        df.head()
        X = df.to_numpy()[:, 0:-1]
        y = df.to_numpy()[:, -1]
        
        self.X = X
        self.y = y
        self.model = LinearRegression(add_bias=True)
        self.model.fit(self.X, self.y)

    def test_coefficients(self):
        np.testing.assert_allclose(self.model.beta, [-2.11943, 0.01280, 0.36484, 0.12540, 0.06669, -0.19453, 0.27829, -0.00196, -0.13579, 0.54782, -0.06806, 0.26881], atol=1e-4)

    def test_residual_standard_error(self):
        rse = self.model.residuals_SE()
        self.assertAlmostEqual(rse, 0.1174, places=4)

    def test_coefficients_standard_error(self):
        cse = self.model.coefficients_SE()
        np.testing.assert_allclose(cse, [0.51904, 0.02586, 0.12936, 0.08971, 0.14303, 0.14779, 0.13826, 0.24001, 0.26725, 0.27725, 0.31485, 0.06213], atol=1e-4)

    def test_coefficients_p_values(self):
        _, p_values = self.model.coefficients_p_values()
        np.testing.assert_allclose(p_values, [0.000290, 0.624272, 0.008291, 0.172102, 0.644293, 0.197735, 0.052883, 0.993536, 0.614972, 0.057121, 0.830269, 0.000146], atol=1e-6)
        
    def test_r_squared(self):
        R, R_adj = self.model.R_squared()
        self.assertAlmostEqual(R, 0.9916, places=4)
        self.assertAlmostEqual(R_adj, 0.9886, places=4) 

    def test_f_statistic(self):
        F, p = self.model.F_score()
        self.assertAlmostEqual(F, 332.9, places=1)   # Should be very large (perfect fit)
        self.assertLess(p, 2.2e-16)      # p-value should be tiny

    def test_residual_stats(self):
        quartiles = self.model.residual_stats()
        np.testing.assert_allclose(quartiles, [-0.22123, -0.06155, -0.01055, 0.05045, 0.26079], atol=1e-4)

if __name__ == '__main__':
    unittest.main()