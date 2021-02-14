from scipy import linalg, sparse
import numpy as np
from unittest import TestCase
import timeit
from matplotlib import pyplot as plt


class TestMatrixInverseMethods(TestCase):
    a, b, x = None, None, None

    def setup(self):
        size = 500
        a_gen = np.random.random((size, size)) * 1E-7
        a = (a_gen + a_gen.T)/2
        k = np.array([np.ones(size-1), np.ones(size)*5, np.ones(size-1)])
        offset = [-1, 0, 1]
        a_bulk = sparse.diags(k, offset).toarray() * 10 * 1E7
        a *= np.ones((size, size)) + a_bulk
        a[:50] = np.random.random((50, size)) * 1E-15
        a[:, :50] = np.random.random((size, 50)) * 1E-15
        x = np.random.random(size) * 20
        b = np.matmul(a, x)
        b[:50] = np.random.random(50) * 1E-15
        self.a, self.b, self.x = a, b, x

    def calculate_pinvh(self):
        ainv = linalg.pinvh(self.a)
        x1 = np.matmul(ainv, self.b)
        return self.x, x1

    def test_time_and_assert_pinvh(self):
        # Timeit
        number = 5
        print("Timing it")
        time = timeit.repeat(lambda: self.calculate_pinvh(), lambda: self.setup(), number=number, repeat=10)
        print(f"Execution ms: {np.average(np.array(time))/number*1000}")

        for i in range(5):
            x, x1 = self.calculate_pinvh()
            plt.plot(x, color='green')
            plt.plot(x1, color='blue')
            plt.show()
            plt.plot(x - x1, color='red')
            plt.show()
            print("Logging")
            print(np.amax(x - x1))

    def calculate_lstsq(self):
        x1 = linalg.lstsq(self.a, self.b)
        return self.x, x1[0]

    def test_time_and_assert_lstqs(self):
        # Timeit
        number = 5
        print("Timing it")
        time = timeit.repeat(lambda: self.calculate_lstsq(), lambda: self.setup(), number=number, repeat=10)
        print(f"Execution ms: {np.average(np.array(time))/number*1000}")

        for i in range(5):
            self.setup()
            x, x1 = self.calculate_lstsq()
            plt.plot(x, color='green')
            plt.plot(x1, color='blue')
            plt.show()
            plt.plot(x - x1, color='red')
            plt.show()
            print("Logging")
            print(np.amax(x - x1))
