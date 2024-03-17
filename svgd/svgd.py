import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from moviepy.video.io.bindings import mplfig_to_npimage
import imageio
from tqdm import tqdm

def gaussian_pdf(x, mu, sigma):
	"""
	:param x: [..., dim]
	:param mu: [dim]
	:param sigma: [dim, dim]
	:return:
	"""
	n = x.shape[-1]
	det = np.linalg.det(sigma)
	const = np.sqrt( (np.pi * 2) ** n ) * np.sqrt(det)
	const = 1 / const

	exp = np.exp(
		-0.5 *
		np.sum(
			((x - mu) @ np.linalg.inv(sigma)) * (x - mu), axis=-1
		)
	)

	p = const * exp
	grad =- p[...,None] * (x - mu) @ np.linalg.inv(sigma)
	return p, grad


def rbf_kernel(x, b=0.5):
	"""
	:param x: [batch, dim]
	:param b:
	:return:
	"""
	l22 = pairwise_distances(x, x, metric='euclidean') ** 2
	k = np.exp(- b * l22)
	grad = k[..., None] * (-2 * b) * np.stack([
		pairwise_distances(x, x, lambda x1, x2: x1[i] - x2[i])
		for i in range(x.shape[-1])
	], axis=-1)
	return k, grad



def example():
	mu1 = np.array([-1.5, -1.5])
	mu2 = np.array([1.5, 1.5])
	sigma1 = np.array([[1, 0], [0, 1]]) * 0.3
	sigma2 = np.array([[1, 0], [0, 1]]) * 0.3

	x1 = np.linspace(-4, 4, 100)
	x2 = np.linspace(-4, 4, 100)
	x1, x2 = np.meshgrid(x1, x2)
	x_plot = np.stack([x1, x2], axis=-1)
	p1, _ = gaussian_pdf(x_plot, mu1, sigma1)
	p2, _ = gaussian_pdf(x_plot, mu2, sigma2)
	p_target = p1/2 + p2/2


	xs = np.random.uniform(-3, 3, (150, 2))
	batch, dim = xs.shape
	gifs = []
	for iter in tqdm(range(1500)):
		if iter % 50 == 0:
			fig = plt.figure()
			plt.contourf(x1, x2, p_target, levels=100, cmap='viridis')
			plt.scatter(xs[:,0], xs[:,1], c='r', s=10)
			plt.colorbar()
			numpy_fig = mplfig_to_npimage(fig)
			gifs.append(numpy_fig)
			plt.close(fig)

		p1, grad_p1 = gaussian_pdf(xs, mu1, sigma1)
		p2, grad_p2 = gaussian_pdf(xs, mu2, sigma2)
		p = p1/2 + p2/2 # [batch]
		grad_p = grad_p1/2 + grad_p2/2  # [batch, dim]
		grad_logp = grad_p / p[:,None] # [batch, dim]

		k, grad_k = rbf_kernel(xs) # [batch, batch], [batch, batch, dim]

		dx = k @ grad_logp / batch + np.mean(grad_k, axis=0) # [batch, dim]

		xs += 0.01 * dx
	imageio.mimsave('svgd.gif', gifs, duration=200, loop=1000)


if __name__ == '__main__':
	example()
	# x = np.random.randn(10, 2)
	# g = np.stack([
	# 	pairwise_distances(x, x, lambda x1, x2: x1[0] - x2[0]),
	# 	pairwise_distances(x, x, lambda x1, x2: x1[1] - x2[1])
	# ], axis=-1)
	# print(g.shape)

