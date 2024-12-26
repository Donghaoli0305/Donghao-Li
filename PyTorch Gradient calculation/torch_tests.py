"""
This file is individualized for NetID dli106.
"""
import numpy as np
import unittest as ut
import torch_code as tc

class TorchTestCase(ut.TestCase):

    def test_mountain1d0(self):
        x = 0.0
        z, dz_dx = tc.mountain1d(x)
        self.assertTrue(np.fabs(3.0 - z) < 0.001)
        self.assertTrue(np.fabs(1.0 - dz_dx) < 0.001)

    def test_mountain1d1(self):
        x = 1.0
        z, dz_dx = tc.mountain1d(x)
        self.assertTrue(np.fabs(2.0 - z) < 0.001)
        self.assertTrue(np.fabs(-6.0 - dz_dx) < 0.001)

    def test_robot0(self):
        t1, t2 = 0.0, 0.0
        z, dz_dt1, dz_dt2 = tc.robot(t1, t2)
        self.assertTrue(np.fabs(73.0000 - z) < 0.001)
        self.assertTrue(np.fabs(30.0000 - dz_dt1) < 0.001)
        self.assertTrue(np.fabs(12.0000 - dz_dt2) < 0.001)

    def test_robot1(self):
        t1, t2 = 0.0, 1.0
        z, dz_dt1, dz_dt2 = tc.robot(t1, t2)
        self.assertTrue(np.fabs(72.0649 - z) < 0.001)
        self.assertTrue(np.fabs(14.3860 - dz_dt1) < 0.001)
        self.assertTrue(np.fabs(-13.7117 - dz_dt2) < 0.001)

    def test_neural_network0(self):
        W1 = np.zeros((1,2), dtype=np.float32)
        W2 = np.zeros((2,3), dtype=np.float32)
        W3 = np.zeros((3,4), dtype=np.float32)
        output = tc.neural_network(W1, W2, W3)

        y, e = output[:2]
        self.assertTrue(np.fabs(0.0000 - y) < 0.001)
        self.assertTrue(np.fabs(1.0000 - e) < 0.001)

        de_dW_actual = output[2:]
        de_dW_expected = [
            np.array([
                [1.5232,-1.5232]]),
            np.array([
                [0.0000,0.0000,0.0000],
                [0.0000,0.0000,0.0000]]),
            np.array([
                [-0.0000,-0.0000,-0.0000,-0.0000],
                [-0.0000,-0.0000,-0.0000,-0.0000],
                [-0.0000,-0.0000,-0.0000,-0.0000]]),
        ]
        for (actual, expected) in zip(de_dW_actual, de_dW_expected):
            self.assertTrue(np.fabs(actual - expected).max() < 0.001)

    def test_neural_network1(self):
        W1 = np.ones((1,2), dtype=np.float32)
        W2 = np.ones((2,3), dtype=np.float32)
        W3 = np.ones((3,4), dtype=np.float32)
        output = tc.neural_network(W1, W2, W3)

        y, e = output[:2]
        self.assertTrue(np.fabs(-1.9555 - y) < 0.001)
        self.assertTrue(np.fabs(8.7353 - e) < 0.001)

        de_dW_actual = output[2:]
        de_dW_expected = [
            np.array([
                [5.9063,5.6532]]),
            np.array([
                [0.0093,0.0093,0.0093],
                [0.4880,0.4880,0.4880]]),
            np.array([
                [0.0253,0.0253,0.0253,0.0253],
                [0.0253,0.0253,0.0253,0.0253],
                [0.0253,0.0253,0.0253,0.0253]]),
        ]
        for (actual, expected) in zip(de_dW_actual, de_dW_expected):
            self.assertTrue(np.fabs(actual - expected).max() < 0.001)


if __name__ == "__main__":    
    
    test_suite = ut.TestLoader().loadTestsFromTestCase(TorchTestCase)
    res = ut.TextTestRunner(verbosity=2).run(test_suite)
    num, errs, fails = res.testsRun, len(res.errors), len(res.failures)
    print("score: %d of %d (%d errors, %d failures)" % (num - (errs+fails), num, errs, fails))
    


