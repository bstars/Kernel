

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances

def kernel_rbf(x1, x2, sigma=1.0):
	return np.exp(-((x1 - x2) ** 2) / (2 * (sigma ** 2)))


def generate(xs):
	return kernel_rbf(xs, 1, sigma=0.5) + 1.5 * kernel_rbf(xs, -1, sigma=0.5) + 0.05 * np.random.randn(*xs.shape)


class KernelRegression():
	# 1D kernel regression
	def __init__(self, kernel_func, lamb=0.01):
		self.kernel_func = kernel_func
		self.lamb = lamb
		self.alpha = None
		self.xs = None

	def predict(self, x):
		"""
		:param x: [M,]
		:return: [M,]
		"""
		K = pairwise_distances(x[:,None], self.xs[:,None], metric=self.kernel_func)
		return K @ self.alpha


	def fit(self, x, y, ls_alpha=0.05, ls_beta=0.5):
		"""
		:param x: [N,]
		:param y: [N,]
		:return:
		"""
		N = len(x)
		self.alpha = np.zeros_like(x)
		self.xs = x
		K = pairwise_distances(x[:,None], x[:,None], metric=self.kernel_func)
		xtest = np.linspace(-2, 2, 300)

		def loss_aux(alpha):
			pred = K @ alpha
			return np.mean((pred - y) ** 2) + self.lamb * alpha @ K @ alpha



		for _ in range(1000):
			fx = self.predict(x)
			error = fx - y
			loss = np.mean(error ** 2) + self.lamb * self.alpha @ K @ self.alpha
			grad = 2/N * (fx - y) + 2 * self.lamb * self.alpha

			# ||DL(f)||_H^2
			grad2 = 4/N**2 * error @ K @ error \
			        + 4 * self.lamb**2 * self.alpha @ K @ self.alpha \
					+ 8 * self.lamb / N * error @ fx

			# back tracking line search in RKHS
			# while L(f) - ls_alpha * t * ||DL(f)||_H^2 < L(f - t * DL(f))
			t = 1.
			while loss - ls_alpha * t * grad2  < loss_aux(self.alpha - t * grad):
				t = ls_beta * t


			self.alpha = self.alpha - t * grad
			print(t, loss)

			if _ % 20 == 0:
				plt.scatter(x, y, color='blue', alpha=0.5, label='data')
				plt.plot(xtest, self.predict(xtest))
				plt.show()

if __name__ == '__main__':
	xs = np.linspace(-2, 2, 40)
	ys = generate(xs)
	# plt.scatter(xs, ys, color='blue', alpha=0.5)
	# plt.show()

	model = KernelRegression(kernel_func=kernel_rbf, lamb=1e-3)
	model.fit(xs, ys)
	print(model.alpha)