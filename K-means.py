import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
import random
from scipy import spatial
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, Data, number_of_clusters=3, test_size=0.01, random_state=42, number_of_features=3):
        self.Data = Data
        self.number_clusters, self.test_size, self.random_state, self.len_data, self.number_of_features \
            = number_of_clusters, test_size, random_state, len(Data), number_of_features
        self.X, self.y = self.data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.normalize_data, self.normalize_test_data = self.normalize_data()
        self.distance_matrix = None
        self.closestpoints = None
        self.cluster = []

    def data(self):
        # add Error if input self.dimension > No. of features
        X = self.Data.data[:, :self.number_of_features]
        y = self.Data.target
        return X, y

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size,
                                                            random_state=self.random_state)
        return X_train, X_test, y_train, y_test

    def normalize_data(self):
        normalize_train_data, normalize_test_data = MinMaxScaler().fit_transform(
            self.X_train), MinMaxScaler().fit_transform(self.X_test)
        return normalize_train_data, normalize_test_data

    def start_points(self):
        # add more options at start points
        # random_index_from_data1 = np.random.randint(len(self.X_train), size=self.number_clusters) #duplicates start point!
        random_index_from_data = random.sample([i for i in range(len(self.X_train))], self.number_clusters)
        initial_random_points_from_data = self.X_train[random_index_from_data]

        random_points = np.array([np.random.rand(self.number_of_features)] for i in range(self.number_clusters))
        # X_train_without_start_points = np.delete(self.X_train, random_index_from_data, 0)
        # return X_train_without_start_points, random_points_from_data
        return initial_random_points_from_data, random_points

    def Calc_distance_matrix(self, points):
        self.distance_matrix = spatial.distance_matrix(points, self.X_train, p=2)

    def closest_points(self):
        self.closestpoints = np.argmin(self.distance_matrix, axis=0)

    def make_clusters(self):
        for i in range(self.number_clusters):
            self.cluster.append(self.X_train[np.where(self.closestpoints == i)])

    def finding_new_centroid(self):
        new_centroid = []
        for i in range(self.number_clusters):
            new_centroid.append(self.cluster[i].mean(axis=0))
        return new_centroid

    def run(self, number_of_iterations):
        points, points2 = self.start_points()
        print(type(points))
        print(type(points2))
        print(len(points))
        print(len(points2))

        i = 0
        while i < number_of_iterations:
            self.Calc_distance_matrix(points)
            self.closest_points()
            self.make_clusters()
            if i < number_of_iterations - 1:
                points = self.finding_new_centroid()
                self.cluster.clear()
            i += 1
        return points

    def WSS(self, central_points):
        result = []
        for i in range(0, len(central_points)):
            central_points[i] = np.reshape(central_points[i], (-1, len(central_points[i])))
            result.append(spatial.distance_matrix(central_points[i], self.cluster[i], p=2).sum())
        result = np.array(result).sum()
        return result


if __name__ == '__main__':
    dataset = datasets.load_iris()
    # run_K_means = KMeans(dataset)
    number_clusters = []
    elbow = []
    for i in range(1, 8):
        number_clusters.append(i)
        run_K_means = KMeans(dataset, number_of_clusters=i)
        points = run_K_means.run(3)
        elbow.append(run_K_means.WSS(points))
    plt.plot(number_clusters, elbow)
    plt.show()
