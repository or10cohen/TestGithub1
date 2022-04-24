import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets


class KMeans:
    def __init__(self, Data, number_clusters=2, max_distance=200, test_size=0.01, random_state=42, dimension=2):
        self.Data = Data
        self.number_clusters, self.max_distance, self.test_size, self.random_state, self.len_data, self.dimension \
            = number_clusters, max_distance, test_size, random_state, len(Data), dimension
        self.X, self.y = self.data()
        self.X_train,self.X_test, self.y_train, self.y_test = self.split_data()
        self.normalize_data, self.normalize_test_data = self.normalize_data()
        self.X_train_without_start_points, self.random_points_from_data = self.start_points()
        self.distance_vector = self.points_distance()
        self.min_distance = self.min_distance()

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
        random_index_from_data = np.random.randint(len(self.X_train), size=self.number_clusters) #duplicates start point!
        random_points_from_data = self.X_train[random_index_from_data]
        X_train_without_start_points = np.delete(self.X_train, random_index_from_data, 0)
        return X_train_without_start_points, random_points_from_data

    def points_distance(self):
        distance_vector = [[i] for i in range(self.number_clusters)]
        for index, random_points in enumerate(self.random_points_from_data):
            for train_point in self.X_train_without_start_points:
                distance = np.linalg.norm(random_points - train_point)
                distance_vector[index].append(distance)
        return distance_vector

    def min_distance(self):
        min_distance = []
        for i in range(len(self.distance_vector)):
            min_distance.append(np.argmin(self.distance_vector[i]))
        return min_distance

    def update_distance_vector(self):
        min_value = np.argmin(self.min_distance)




if __name__ == '__main__':
    dataset = datasets.load_iris()
    run_K_means = KMeans(dataset)
    print("Or Yosef Cohen")
    # print(len(run_K_means.Data.data))
    # print(len(run_K_means.X_train))
    # print(len(run_K_means.X_test))
    print(len(run_K_means.X_train))
    print(len(run_K_means.X_train_without_start_points))
    print(len(run_K_means.random_points_from_data))
    print(run_K_means.random_points_from_data)

    # print(run_K_means.distance_vector)
    # print(len(run_K_means.distance_vector))
    # print(len(run_K_means.distance_vector[0]))
    # print(len(run_K_means.distance_vector[1]))
    # print(type(run_K_means.distance_vector[0]))
    # print(min(run_K_means.distance_vector[0]))
    # print(min(run_K_means.distance_vector[1]))
    # print(run_K_means.min_distance)
