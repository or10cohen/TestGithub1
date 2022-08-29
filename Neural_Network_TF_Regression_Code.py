import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset=True)
from keras.utils.vis_utils import plot_model
from nnv import NNV




class FirsRegressionNeuralNetwork:
    def __init__(self, X, y, n_epochs=20):
        self.X = X
        self.y = y
        self.n_epochs = n_epochs

    def split_and_normalize_data(self, test_size=0.3, random_state=42):
        X_train, X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                      random_state=random_state)
        # X_train, X_test = MinMaxScaler().fit_transform(X_train), MinMaxScaler().fit_transform(X_test)
        self.scalar = MinMaxScaler()
        self.scalar.fit(X_train)
        self.scalar.fit(X_test)  ##????
        self.X_train, self.X_test = self.scalar.transform(X_train), self.scalar.transform(X_test)

    def create_neural_network(self, No_hidden_layers=3, No_neurons_per_layer=None,
                              activation_per_layer=None):
        ###------------creat NN option1--------------------------
        model1 = tf.keras.Sequential([tf.keras.layers.Dense(4, activation='relu'),
                                      tf.keras.layers.Dense(3, activation='relu'),
                                      tf.keras.layers.Dense(3, activation='relu'),
                                      tf.keras.layers.Dense(1)])  # output layer
        # ###------------creat NN option2-----------------------------
        model2 = tf.keras.Sequential()   ## create  neural
        model2.add(tf.keras.layers.Dense(4, activation='relu')) ##+input_shape(number of fitcher,), bias(True/False)
        model2.add(tf.keras.layers.Dense(4, activation='relu')) #Dense==fully connected(Number of noyrons, activation='activision func'))
        model2.add(tf.keras.layers.Dense(4, activation='relu'))
        model2.add(tf.keras.layers.Dense(1))  # output layer
        ###------------creat NN option3 with loop for Gui-----------------------------
        if activation_per_layer is None:
            activation_per_layer = ['relu' for i in range(No_hidden_layers)]
        if No_neurons_per_layer is None:
            No_neurons_per_layer = [1 for i in range(No_hidden_layers)]
        model3 = tf.keras.Sequential()  ## create  neural
        for i in range(No_hidden_layers):
            model3.add(tf.keras.layers.Dense(No_neurons_per_layer[i], activation=activation_per_layer[i]))
        model3.add(tf.keras.layers.Dense(1))  # output layer
        #------------------------------------------------------------
        self.model1, self.model2, self.model3, self.No_hidden_layers, self.No_neurons_per_layer,\
        self.activation_per_layer = model1, model2, model3, No_hidden_layers, No_neurons_per_layer, activation_per_layer

    def run_model(self, optimizer='rmsprop', loss='mse'):   #loss classification = softmax(נותן לכל נוירון באווטפוט יחס לתשובה הנכונה, הסכום שווה ל1)
        self.model3.compile(optimizer=optimizer, loss=loss)  #optimizer = gradient decsent  , loss = loss function  ++metrices = 'accuracy'
        self.model3.fit(x=self.X_train, y=self.y_train, epochs=self.n_epochs) #epoches = steps/iteraion in gradient decsent for all Data! ++batch_size=32
        ## score = model2.evaluate(x_test, y_test) #++metrices = 'accuracy' on compile
        ##+fit(callback = earlysStopping - stop when 1. loss function no change or somthing else)
        ##+fit(callback = ModelCheck point - save the wihght in the process some time)
        ##+fit(tensorBoard = debbug the neuralNetwork)

    def epochs_graph(self):
        ##---------------------graph epochs--------------------------
        axis_x = [i for i in range(self.n_epochs)]
        axis_y = self.model3.history.history['loss']
        plt.title("Lose Function Per Epoch")
        plt.plot(axis_x, axis_y)
        plt.xlabel("Epochs")
        plt.ylabel("Loss Function")
        plt.savefig('Graph.png')
        # im = Image.open('Graph.png')
        # im.show()
        # ###------------------------0-----------------------------------

        layersList = []
        for i in range(self.No_hidden_layers):
            layersList.append({"title": "hidden\n" + (self.activation_per_layer[i]), "units": self.No_neurons_per_layer[i], "color": "darkBlue"})
        layersList.append( {"title": "output\n(sigmoid)", "units": 1, "color": "darkBlue"},)
        NNV(layersList).render(save_to_file="NN_graph.png")
        # ###------------------------0-----------------------------------
        # plot_model(
        #     self.model3,
        #     to_file = 'graph_model.png',
        #     show_shapes=False,
        #     show_dtype=False,
        #     show_layer_names=True,
        #     rankdir="TB",
        #     expand_nested=False,
        #     dpi=96,
        #     layer_range=None,
        #     show_layer_activations=False
        # )
        ####-------------------------------------------------------------

        # dg = tf.keras.DenseGraph(self.model3)
        # dg.render('DenseGraph.png')

    def predict(self):
        ##------------------------predictions------------------------
        test_predict = self.model3.predict(self.X_test)
        test_predict = pd.Series(test_predict.reshape(len(self.X_test), ))
        pred_df = pd.DataFrame(self.y_test, columns=['Test True Y'])
        pred_df = pd.concat([pred_df, test_predict], axis=1)
        pred_df.columns = ['Test True Y', 'Model predict']
        print(pred_df)
        ###------------------------predictions on new data without label------------------------
        self.new_gem = [[998, 1000]]
        self.new_gem = self.scalar.transform(self.new_gem)  # scale by the X_train
        print(self.new_gem)
        print(Fore.MAGENTA + 'predictions on new data without label', self.model3.predict(self.new_gem))

    def accuracy(self):
        #use the metrices you use on model3.compile
        #score = model3.evolaite()
        pass

    def save_and_load_model(self):
        ###-----------------------save and load your model----------------------------------------
        self.model3.save('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\')
        loaded_model = tf.keras.models.load_model('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\')
        loaded_model.predict(self.new_gem)
        print(Fore.MAGENTA + 'new model predictions on new data without label', loaded_model.predict(self.new_gem))
        if loaded_model.predict(self.new_gem) == self.model3.predict(self.new_gem):
            print(Fore.GREEN + 'the loaded_model = save model. good save and load!')
        else:
            print(Fore.GREEN + 'error load or save model!')


if __name__ == '__main__':
    df = pd.read_csv('DATA/fake_reg.csv')
    X = df[['feature1', 'feature2']].values
    y = df['price'].values
    run = FirsRegressionNeuralNetwork(X, y)
    run.split_and_normalize_data()
    run.create_neural_network()
    run.run_model()
    run.epochs_graph()
    run.predict()
    run.save_and_load_model()
