import scipy.io
import colorama
import numpy as np
from colorama import Fore, Back, Style
colorama.init(autoreset = True)
import pandas as pd
import matplotlib.pyplot as plt

class TSNE():
    def __init__(self, data_data, data_label):
        self.X = data_data.T
        self.y = data_label.T

    def __str__(self):
        print(Fore.GREEN + 'data shape & label shape:\n', self.X.shape, self.y.shape)
        print(Fore.GREEN + 'type(data & label):\n', type(self.X), type(self.y))
        print(self.df)
        print(Fore.GREEN + 'Size of the dataframe:\n', self.df.shape)
        print(Fore.GREEN + 'type(dataframe):\n', type(self.df))

    def normalizeData(self):
        self.X = self.X / 255

    def dataFrameConvert(self):
        self.feat_cols = ['pixel' + str(i) for i in range(self.X.shape[1])]
        self.df = pd.DataFrame(self.X, columns=self.feat_cols)
        self.df['y'] = self.y
        self.df['label'] = self.df['y'].apply(lambda i: str(i))
        # X, y = None, None

    def plotDataFrameExsample(self):
        np.random.seed(42)
        rndperm = np.random.permutation(self.df.shape[0])
        plt.gray()
        fig = plt.figure(figsize=(16, 7))
        for i in range(0, 15):
            ax = fig.add_subplot(3, 5, i + 1, title="Digit: {}".format(str(self.df.loc[rndperm[i], 'label'])))
            ax.matshow(self.df.loc[rndperm[i], self.feat_cols].values.reshape((28, 28)).astype(float))
        plt.show()

if __name__ == '__main__':
    mnist = scipy.io.loadmat('mnist-original.mat')
    data = mnist['data']
    label = mnist['label']

    Try_TSNE = TSNE(data_data=data, data_label=label)
    Try_TSNE.normalizeData()
    Try_TSNE.dataFrameConvert()
    Try_TSNE.plotDataFrameExsample()
    Try_TSNE.__str__()
