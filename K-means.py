import numpy as np
import numpy.random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets


class K_means:
    def __init__(self, Data, number_clusters=2, max_distance=200, test_size=0.01, random_state=42, dimension=2):
        self.Data = Data
        self.number_clusters, self.max_distance, self.test_size, self.random_state, self.len_data, self.dimension \
            = number_clusters, max_distance, test_size, random_state, len(Data), dimension
        self.X, self.y = self.data()
        self.X_train,self.X_test, self.y_train, self.y_test = self.split_data()
        self.normalize_data, self.normalize_test_data = self.normalize_data()

    def data(self):
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
        X_train_array = np.array(self.X_train)
        numpy.random.choice(X_train_array, size=self.number_clusters, replace=False)

if __name__ == '__main__':
    dataset = datasets.load_iris()
    run_K_means = K_means(dataset)
    print("Or Yosef Cohen")
    print(len(run_K_means.Data.data))
    print(len(run_K_means.X_train))
    print(len(run_K_means.X_test))
    print(run_K_means.X_train)