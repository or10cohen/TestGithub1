import numpy as np
import imageio
import scipy as sc
from sklearn import datasets
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.colors as colors
import matplotlib.animation as ani
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class GMM:
    def __init__(self, data, number_of_clusters=3, n_epochs=30, epsilon_likelihood=1):
        self.X, self.n_clusters, self.n_epochs, self.epsilon_likelihood = \
            data, number_of_clusters, n_epochs, epsilon_likelihood
        self.totals = None
        self.clusters = []
        self.gamma_nk = []
    # @staticmethod
    def gaussian(self, mu, cov):
        # Gaussian_P_formula = Image.open("C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\Gaussian_P_formula.PNG")
        # Gaussian_P_formula.show()
        n = self.X.shape[1]  # number of columns
        diff = (self.X - mu).T
        return np.diagonal(1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5) * np.exp(
            -0.5 * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff))).reshape(-1, 1)
    def initialize_clusters(self):
        # We use the KMeans centroids to initialise the GMM
        kmeans = KMeans(self.n_clusters).fit(self.X)
        mu_k = kmeans.cluster_centers_
        cov_matrix = np.cov(X.T)

        for i in range(self.n_clusters):
            self.clusters.append({
                'pi_k': 1.0 / self.n_clusters,  ##1\3 1\3 1\3 at start if you have 3 clusters
                'mu_k': mu_k[i],
                'cov_k': cov_matrix  # need to use mu_k----Oren you use **data** for the covariance matrix----
            })
        return self.clusters
    def expectation_step(self):  # calculating responsibility matrix
        # Gamma_nk_formula = Image.open("C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\Gamma_nk_formula.PNG")
        # Gamma_nk_formula.show()
        N = self.X.shape[0]  # number of rows
        K = len(self.clusters)
        self.totals = np.zeros((N, 1), dtype=np.float64)
        self.gamma_nk = np.zeros((N, K), dtype=np.float64)

        for k, cluster in enumerate(self.clusters):
            pi_k = cluster['pi_k']
            mu_k = cluster['mu_k']
            cov_k = cluster['cov_k']

            self.gamma_nk[:, k] = (pi_k * self.gaussian(mu_k, cov_k)).ravel()

        self.totals = np.sum(self.gamma_nk, 1)
        self.gamma_nk /= np.expand_dims(self.totals, 1)
    def maximization_step(self):
        N = float(self.X.shape[0])

        for k, cluster in enumerate(self.clusters):
            gamma_k = np.expand_dims(self.gamma_nk[:, k], 1)
            # Nk_formula = Image.open("C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\N_k_formula.PNG")
            # Nk_formula.show()
            N_k = np.sum(gamma_k, axis=0)
            # pi_k_formula = Image.open("C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\pi_k_formula.PNG")
            # pi_k_formula.show()
            pi_k = N_k / N
            # mu_k_formula = Image.open("C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\mu_k_formula.PNG")
            # mu_k_formula.show()
            mu_k = np.sum(gamma_k * self.X, axis=0) / N_k
            # cov_k_formula = Image.open("C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\mu_k_formula.PNG")
            # cov_k_formula.show()
            cov_k = (gamma_k * (self.X - mu_k)).T @ (self.X - mu_k) / N_k

            cluster['pi_k'] = pi_k
            cluster['mu_k'] = mu_k
            cluster['cov_k'] = cov_k
    def get_likelihood(self):
        # likelihoods_formula = Image.open("C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\likelihoods_formula.PNG")
        # likelihoods_formula.show()
        sample_likelihoods = np.log(self.totals)
        return np.sum(sample_likelihoods), sample_likelihoods
    def plot_likelihood(self):
        plt.figure(figsize=(10, 10))
        plt.title('Log-Likelihood')
        plt.plot(np.arange(1, self.n_epochs + 1), self.likelihoods)
        plt.show()
        plt.savefig("plot_likelihood.png")
    def create_cluster_animation(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        colorset1 = ['blue', 'red', 'black']
        images = []

        for j, clusters in enumerate(self.history):
            idx = 0
            if j % 3 != 0:
                continue

            plt.cla()

            for cluster in self.clusters:
                mu = cluster['mu_k']
                cov = cluster['cov_k']

                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                order = eigenvalues.argsort()[::-1]
                eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
                vx, vy = eigenvectors[:, 0][0], eigenvectors[:, 0][1]
                theta = np.arctan2(vy, vx)

                color = colors.to_rgba(colorset1[idx])

                for cov_factor in range(1, 4):
                    ell = Ellipse(xy=mu, width=np.sqrt(eigenvalues[0]) * cov_factor * 2,
                                  height=np.sqrt(eigenvalues[1]) * cov_factor * 2, angle=np.degrees(theta), linewidth=2)
                    ell.set_facecolor((color[0], color[1], color[2], 1.0 / (cov_factor * 4.5)))
                    ax.add_artist(ell)

                ax.scatter(cluster['mu_k'][0], cluster['mu_k'][1], c=colorset1[idx], s=1000, marker='+')
                idx += 1

            for i in range(X.shape[0]):
                ax.scatter(X[i, 0], X[i, 1], c=colorset1[np.argmax(self.scores[i])], marker='o')

            fig.canvas.draw()

            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            images.append(image)

        kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
        imageio.mimsave('./gmm.gif', images, fps=1)
        plt.show(Image.open('gmm.gif').convert('RGB'))
    def train_gmm(self):
        self.clusters = self.initialize_clusters()
        self.likelihoods = np.zeros((self.n_epochs,))
        scores = np.zeros((X.shape[0], self.n_clusters))
        self.history = []
        stop = 100
        for i in range(self.n_epochs):
            clusters_snapshot = []

            # This is just for our later use in the graphs
            for cluster in self.clusters:
                clusters_snapshot.append({
                    'mu_k': cluster['mu_k'].copy(),
                    'cov_k': cluster['cov_k'].copy()
                })

            self.history.append(clusters_snapshot)

            self.expectation_step()
            self.maximization_step()

            likelihood, self.sample_likelihoods = self.get_likelihood()
            self.likelihoods[i] = likelihood

            print('Epoch: ', i + 1, 'Likelihood: ', likelihood)

        self.scores = np.log(self.gamma_nk)



if __name__ == '__main__':
    data = datasets.load_iris()
    X = data.data
    print(X.shape)
    print(np.cov(X.T).shape)
    gmm = GMM(X, number_of_clusters=3)
    gmm.train_gmm()
    gmm.plot_likelihood()
    # gmm.create_cluster_animation()
