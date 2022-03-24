import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset = True)
from PIL import Image



class svm:
    def __init__(self, chose_dataset, test_size=0.33, random_state=42, kernel='linear', c=1.0):
        self.data_set = chose_dataset
        self.test_size, self.random_state, self.kernel, self.C = test_size, random_state, kernel, c
        self.normalize_X_train, self.y = self.data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.normalize_X_train, self.normalize_X_test = self.normalize_data()
        self.clf = self.fit()
        self.predict_on_model, self.confusion_matrices, self.confusion_matrix_displays, self.classification_reports \
                                                                                                     = self.predict()


    def __str__(self):
        print(Fore.GREEN + '\ndataset.keys()\n', self.data_set.keys())
        # print(Fore.GREEN + '\ndataset.filename\n', self.data_set.filename)
        print(Fore.GREEN + '\ndataset.data[:5, :5]\n', self.data_set.data[:5, :5])
        print(Fore.GREEN + '\ndataset.data.shape:\n', self.data_set.data.shape)
        print(Fore.GREEN + '\ntype(self.data_set.data):\n', type(self.data_set.data))
        print(Fore.GREEN + '\ndataset.target[:20]\n', self.data_set.target[:20])
        print(Fore.GREEN + '\nnp.unique(self.data_set.target)\n', np.unique(self.data_set.target))
        print(Fore.GREEN + '\ndataset.target_names\n', self.data_set.target_names)
        print(Fore.GREEN + '\ndataset.target.shape\n', self.data_set.target.shape)

    def data(self):
        X = self.data_set.data[:,:2]
        y = self.data_set.target
        return X, y

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.normalize_X_train, self.y, test_size=self.test_size,
                                                            random_state=self.random_state)
        return X_train, X_test, y_train, y_test

    def normalize_data(self):
        normalize_train_data, normalize_test_data = MinMaxScaler().fit_transform(
            self.X_train), MinMaxScaler().fit_transform(self.X_test)
        return normalize_train_data, normalize_test_data

    def fit(self):
        clf = SVC(C=self.C, kernel=self.kernel)
        clf.fit(self.normalize_X_train, self.y_train)
        return clf

    def predict(self):
        predict_on_model = self.clf.predict(self.normalize_X_test)
        confusion_matrices = confusion_matrix(self.y_test, predict_on_model)
        confusion_matrix_displays = ConfusionMatrixDisplay(confusion_matrix=confusion_matrices, \
                                    display_labels=self.clf.classes_)
        classification_reports = classification_report(self.y_test, predict_on_model)
        return predict_on_model, confusion_matrices, confusion_matrix_displays, classification_reports


    def plot_2d(self, table_parameter_min_x=0, table_parameter_max_x=0, table_parameter_min_y=0, \
                table_parameter_max_y=0):
        clf = self.clf

        x_min, x_max =  self.normalize_X_train[:, 0].min() - 1, self.normalize_X_train[:, 0].max() + 1
        y_min, y_max =  self.normalize_X_train[:, 1].min() - 1, self.normalize_X_train[:, 1].max() + 1
        # get the separating hyperplane
        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(x_min, x_max)
        yy = a * xx - (clf.intercept_[0]) / w[1]
        # plot the parallels to the separating hyperplane that pass through the
        # support vectors (margin away from hyperplane in direction
        # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
        # 2-d.
        margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
        yy_down = yy - np.sqrt(1 + a ** 2) * margin
        yy_up = yy + np.sqrt(1 + a ** 2) * margin

        # plot the line, the points, and the nearest vectors to the plane
        plt.figure(1, figsize=(4, 3))
        plt.clf()
        plt.plot(xx, yy, "k-")
        plt.plot(xx, yy_down, "k--")
        plt.plot(xx, yy_up, "k--")

        plt.scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=80,
            facecolors="none",
            zorder=10,
            edgecolors="k",
            cmap=cm.get_cmap("RdBu"),
        )
        plt.scatter(
            self.normalize_X_train[:, 0], self.normalize_X_train[:, 1], c=self.y_train, zorder=10, cmap=cm.get_cmap("RdBu"), edgecolors="k"
        )

        plt.axis("tight")


        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf.decision_function(xy).reshape(XX.shape)
        # Put the result into a contour plot
        plt.contourf(XX, YY, Z, cmap=cm.get_cmap("RdBu"), alpha=0.5, linestyles=["-"])
        plt.xlim(x_min + table_parameter_min_x * x_min, x_max - table_parameter_max_x * x_max)
        plt.ylim(y_min + table_parameter_min_y * y_min, y_max - table_parameter_max_y * y_max)
        plt.xticks(())
        plt.yticks(())

        plt.show()


if __name__ == '__main__':
    are_dataset = datasets.load_breast_cancer()
    run_svm = svm(are_dataset)
    run_svm.plot_2d()
    print(Fore.RED + '\n', run_svm.__str__())
    print(Fore.RED + '\nrun_svm.predict_on_model\n', run_svm.predict_on_model)
    print(Fore.RED + '\nrun_svm.predict_on_model.shape\n', run_svm.predict_on_model.shape)
    print(Fore.RED + '\nrun_svm.y_test\n', run_svm.y_test)
    print(Fore.RED + '\nrun_svm.y_test.shape\n', run_svm.y_test.shape)
    print(Fore.RED + '\nconfusion_matrices\n', run_svm.confusion_matrices)
    # z = run_svm.confusion_matrices
    # print('\nTotal label 0: {} \nTrue: {}  False: {} \n\nTotal label 1: {} \nTrue: {}  False: {}' .format(z[0][0] + z[0][1], \
    # z[0][0], z[0][1], z[1][0] + z[1][1], z[1][1], z[1][0]))
    print(Fore.RED + '\nclassification_reports\n', run_svm.classification_reports)
    image = Image.open("SVM_confusion_matrix_displays.png")
    # Image._show(image)