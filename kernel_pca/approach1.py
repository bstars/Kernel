import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_moons, make_circles
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.decomposition import KernelPCA


class GaussianPCAApproach1():
	def __init__(self, n_components=2, sigma=1.0):
		self.alpha = None
		self.n_components = n_components
		self.sigma = sigma

	def fit(self, X):
		self.X = X
		K = pairwise_kernels(X, X, 'rbf', gamma=self.sigma)
		Kbar = K - K.mean(axis=0, keepdims=True)
		Kbar = Kbar.T @ Kbar

		# solve the constrained optimization problem
		K_evals, K_evecs = np.linalg.eigh(K)
		K_evals = np.real(K_evals) + 1e-5
		K_evecs = np.real(K_evecs)
		K05inv = (K_evecs * np.sqrt(1 / K_evals)) @ K_evecs.T
		evals, evecs = np.linalg.eig(K05inv @ Kbar @ K05inv)
		evals = np.real(evals)
		evecs = np.real(evecs)
		idx = np.argsort(evals)[::-1]
		evecs = evecs[:, idx]
		self.alpha = K05inv @ evecs[:, :self.n_components]

	def project(self, Z):

		K = pairwise_kernels(self.X, Z, 'rbf', gamma=self.sigma)

		# coord = []
		# for i in range(self.n_components):
		# 	coord.append(self.alpha[:, i] @ K)
		# return np.stack(coord, axis=0).T
		coord = self.alpha.T @ K
		return coord.T

if __name__ == '__main__':
	X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

	fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
	ax1.scatter(X[y == 0, 0], X[y == 0, 1], color='red', alpha=0.5)
	ax1.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', alpha=0.5)
	ax1.set_title('original data')

	gpca = GaussianPCAApproach1(n_components=2, sigma=15.0)
	gpca.fit(X)
	# Z = gpca.alpha
	Z = gpca.project(X)
	ax2.scatter(Z[y == 0, 0], Z[y == 0, 1], color='red', alpha=0.5)
	ax2.scatter(Z[y == 1, 0], Z[y == 1, 1], color='blue', alpha=0.5)
	ax2.set_title('Approach 1')


	kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15.0)
	Z = kpca.fit_transform(X)
	ax3.scatter(Z[y == 0, 0], Z[y == 0, 1], color='red', alpha=0.5)
	ax3.scatter(Z[y == 1, 0], Z[y == 1, 1], color='blue', alpha=0.5)
	ax3.set_title('sklearn Kernel PCA')
	plt.show()

