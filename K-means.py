import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
import random
from scipy import spatial

class KMeans:
    def __init__(self, Data, number_clusters=3, max_distance=200, test_size=0.01, random_state=42, dimension=3):
        self.Data = Data
        self.number_clusters, self.max_distance, self.test_size, self.random_state, self.len_data, self.dimension \
            = number_clusters, max_distance, test_size, random_state, len(Data), dimension
        self.X, self.y = self.data()
        self.X_train,self.X_test, self.y_train, self.y_test = self.split_data()
        self.normalize_data, self.normalize_test_data = self.normalize_data()
        self.random_points_from_data = self.start_points()
        self.distance_matrix = self.distance_matrix()
        self.who_more_close_at_distance_matrix_per_column = self.who_more_close()
        self.cluster = self.make_clusters()
        self.mean_values = self.finding_new_centroid()

    def data(self):
        # add Error if input self.dimension > No. of features
        X = self.Data.data[:, :self.dimension]
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
        random_points_from_data = self.X_train[random_index_from_data]
        # X_train_without_start_points = np.delete(self.X_train, random_index_from_data, 0)
        # return X_train_without_start_points, random_points_from_data
        return random_points_from_data

    def distance_matrix(self, points):
        distance_matrix = spatial.distance_matrix(points, self.X_train, p=2)
        return distance_matrix

    def who_more_close(self):
        who_more_close_at_distance_matrix_per_column = np.argmin(self.distance_matrix, axis=0)
        return who_more_close_at_distance_matrix_per_column

    def make_clusters(self):
        clusters = []
        for i in range(self.number_clusters):
            clusters.append(self.X_train[np.where(self.who_more_close_at_distance_matrix_per_column == i)])
        return clusters

    def finding_new_centroid(self):
        mean_values = []
        for i in range(self.number_clusters):
            mean_values.append(self.cluster[i].mean(axis=0))
        return mean_values

    def run(self, number_of_iterations):
        points = self.random_points_from_data()
        i = 0
        while i < number_of_iterations:
            self.distance_matrix(points)
            self.who_more_close()
            self.make_clusters()
            points = self.finding_new_centroid()
            i += 1


if __name__ == '__main__':
    dataset = datasets.load_iris()
    print(dataset.keys())
    run_K_means = KMeans(dataset)
    print(type(run_K_means.X_train_without_start_points))
    # print(run_K_means.distance_matrix)
    print(run_K_means.mean_values)


    # print("Or Yosef Cohen")
    # # print(len(run_K_means.Data.data))
    # # print(len(run_K_means.X_train))
    # # print(len(run_K_means.X_test))
    # # print(len(run_K_means.X_train))
    # # print(len(run_K_means.X_train_without_start_points))
    # # print(len(run_K_means.random_points_from_data))
    # print(run_K_means.random_points_from_data)
    #
    # print(run_K_means.distance_vector[0])
    # print(run_K_means.distance_vector[1])
    # print(run_K_means.min_distance_index)
    # print(run_K_means.distance_vector[0][run_K_means.min_distance_index[0]])
    # print(run_K_means.distance_vector[1][run_K_means.min_distance_index[1]])
    # # print(run_K_means.distance_vector[0].sort())
    # print(run_K_means.distance_vector[0])
    # # print(run_K_means.distance_vector[1].sort())
    # print(run_K_means.distance_vector[1])
    #
    # # print(len(run_K_means.distance_vector))
    # # print(len(run_K_means.distance_vector[0]))
    # # print(len(run_K_means.distance_vector[1]))
    # # print(type(run_K_means.distance_vector[0]))
    # # print(min(run_K_means.distance_vector[0]))
    # # print(min(run_K_means.distance_vector[1]))
    # # print(run_K_means.min_distance)
