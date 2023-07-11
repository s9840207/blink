import unittest

import numpy as np

from blink import fitting


class TestFitting(unittest.TestCase):
    def test_functional(self):
        # Jack has a x, y dataset. He want to fit the data to a straight line.
        x = np.linspace(0, 100, 1000)
        y = 2 * x + 1

        # Jack calls the main function in the fitting module, and gets a return
        # value called result, which is a dict.
        result = fitting.main(x, y)
        self.assertIsInstance(result, dict)

        # Jack looks into the dict, and he finds keys called 'slope',
        # 'intercept', 'r_squared'. The corresponding values are the fitting
        # result.
        self.assertIn("slope", result)
        self.assertIn("intercept", result)
        self.assertIn("r_squared", result)

    def test_check_x_y_equal_length(self):
        x = [1, 2, 3]
        y = [4, 5]
        with self.assertRaises(ValueError, msg="Unequal x and y length"):
            fitting.main(x, y)

    def test_fit_linear(self):
        x = [1, 2, 3]
        y = [4, 5, 6]
        return_value = fitting.fit_linear(x, y)
        self.assertIsInstance(return_value, tuple)
        self.assertEqual(len(return_value), 2)
        popt, pcov = return_value
        self.assertEqual(len(popt), 2)
        self.assertEqual(pcov.shape, (2, 2))

    def test_calculate_linear_r_squared(self):
        x_list = [
            [1, 2, 3],
            [1, 2, 3],
            [1, 0, -1, 0],
        ]
        y_list = [
            [1, 2, 3],
            [-1, -2, -3],
            [0, 1, 0, -1],
        ]
        answers = [1, 1, 0]
        for x, y, answer in zip(x_list, y_list, answers):

            popt, _ = fitting.fit_linear(x, y)
            r_squared = fitting.calculate_linear_r_squared(x, y, popt)
            self.assertLessEqual(r_squared, 1)
            self.assertGreaterEqual(r_squared, 0)
            self.assertEqual(r_squared, answer)


if __name__ == "__main__":
    unittest.main()
