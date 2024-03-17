import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_moons, make_circles
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.decomposition import KernelPCA


class GaussianPCAApproach2():
	def __init__(self, n_components=2, sigma=1.0):
		self.alpha = None
		self.n_components = n_components
		self.sigma = sigma

	def fit(self, X):
		N, _ = X.shape
		self.X = X
		K = pairwise_kernels(X, X, 'rbf', gamma=self.sigma)
		K = K - np.mean(K, axis=0, keepdims=True) - np.mean(K, axis=1, keepdims=True) + np.mean(K)

		# one_n = np.ones((N, N)) / N
		# K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

		evals, evecs = np.linalg.eigh(K)
		evals = np.real(evals)
		evecs = np.real(evecs)
		idx = np.argsort(evals)[::-1]
		evecs = evecs[:, idx]
		self.alpha = evecs[:, :self.n_components]

	def project(self, Z):
		"""
		:param Z: [batch, dim]
		:return:
		"""
		coord = []
		K = pairwise_kernels(self.X, Z, 'rbf', gamma=self.sigma)
		# K = pairwise_distances(self.X, Z, metric=rbf_kernel)
		Kbar = K - np.mean(K, axis=0, keepdims=True) - np.mean(K, axis=1, keepdims=True) + np.mean(K)

		coord = self.alpha.T @ K
		return coord.T

if __name__ == '__main__':
	X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

	fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
	ax1.scatter(X[y == 0, 0], X[y == 0, 1], color='red', alpha=0.5)
	ax1.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', alpha=0.5)
	ax1.set_title('original data')

	gpca = GaussianPCAApproach2(n_components=2, sigma=15.0)
	gpca.fit(X)
	# Z = gpca.alpha
	Z = gpca.project(X)
	ax2.scatter(Z[y == 0, 0], Z[y == 0, 1], color='red', alpha=0.5)
	ax2.scatter(Z[y == 1, 0], Z[y == 1, 1], color='blue', alpha=0.5)
	ax2.set_title('Approach 2')


	kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15.0)
	Z = kpca.fit_transform(X)
	ax3.scatter(Z[y == 0, 0], Z[y == 0, 1], color='red', alpha=0.5)
	ax3.scatter(Z[y == 1, 0], Z[y == 1, 1], color='blue', alpha=0.5)
	ax3.set_title('sklearn Kernel PCA')
	plt.show()