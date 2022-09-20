import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import CNN_Convolution_Neural_Network
from PIL import Image
# from dash import html
# import dash_bootstrap_components as dbc
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset=True)
st.set_page_config(layout="wide")
from tensorflow.keras import losses
import tensorflow as tf

def main():
##----------------------------------------------------------------------------------------------------------------------
##----------------------------------------------main page---------------------------------------------------------------
##----------------------------------Convolution Neural Network - by Oren Niazov & Or Yosef Cohen------------------------
##----------------------------------------------------------------------------------------------------------------------
    global Run_Function, Dataset
    st.title('**Convolotion Neural Network - by Oren Niazov & Or Yosef Cohen**')

    if st.button('Run Function'):
        st.write('Running Function')
        Run_Function = 'Run Function'
    else:
        st.write('Press run function to start')
        Run_Function = 'Dont Run Function'
##---------------------------------------------------sidebar------------------------------------------------------------
##-------------------------------------------------choose Dataset-------------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------
    with st.sidebar:
        st.sidebar.title('_Input Parameters_')
        st.title('choose Datatype&Dataset')
        Dataset = st.selectbox(
            'choose dataset',
            ('Mnist', 'CIFAR10'))

        if Dataset == 'Mnist':
            Dataset = tf.keras.datasets.fashion_mnist.load_data()  # N*28*28 (grayscale)
        elif Dataset == 'CIFAR10':
            Dataset = tf.keras.datasets.cifar10.load_data()  # N*32*32*3 (color image)
        else:
            print('Dataset are not Mnist or CIFAR10')
##----------------------------------------------sidebar-----------------------------------------------------------------
##----------------------------------------create_neural_network---------------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------



##------------------------------------------------sidebar---------------------------------------------------------------
##----------------------------------------------- run_model-------------------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------
        st.title('parameters for function: run_model')
        optimizer = st.selectbox(
            'choose optimizer',
            ('rmsprop','SGD', 'adam', 'adamax', 'Nadam'))

        loss = st.selectbox(
            'choose loss function',
            ('SparseCategoricalCrossentropy[class]','BinaryCrossentropy[class]', 'CategoricalCrossentropy[class]', 'Poisson[class]',
             'MeanSquaredError[reg]', 'MeanAbsoluteError[reg]', 'MeanAbsolutePercentageError[reg]','MeanSquaredLogarithmicError[reg]'))

        if loss == 'BinaryCrossentropy[class]':
            loss = losses.BinaryCrossentropy()
        elif loss == 'CategoricalCrossentropy[class]':
            loss = losses.CategoricalCrossentropy()
        elif loss == 'SparseCategoricalCrossentropy[class]':
            loss = losses.SparseCategoricalCrossentropy()
        elif loss == 'Poisson[class]':
            loss = losses.Poisson()
        elif loss == 'MeanSquaredError[reg]':
            loss = losses.MeanSquaredError()
        elif loss == 'MeanAbsoluteError[reg]':
            loss = losses.MeanAbsoluteError()
        elif loss == 'MeanAbsolutePercentageError[reg]':
            loss = losses.MeanAbsolutePercentageError()
        elif loss == 'MeanSquaredLogarithmicError[reg]':
            loss = losses.MeanSquaredLogarithmicError()

        #st.write('loss', loss)
        n_epochs = st.number_input('No of epochs:', value=5)
        #st.write('n_epochs:', n_epoch)
        batch_size = st.number_input('batch_size:', value=32)
        # st.write('batch_size:', batch_size)

##---------------------------------------------------sidebar------------------------------------------------------------
##-----------------------------------------download files you want to share---------------------------------------------
##----------------------------------------------------------------------------------------------------------------------
        with open("CNN_Convolution_Neural_Network.py") as file:
            btn = st.download_button(
                label="Download CNN_Convolution_Neural_Network Python Resources File",
                data=file,
            )

        with open("Gui_CNN_Convolution_Neural_Network.py") as file:
            btn = st.download_button(
                label="Download Gui_CNN_Convolution_Neural_Network Python Resources File",
                data=file,
            )

##----------------------------------------------------------------------------------------------------------------------
##----------------------------------------------main page---------------------------------------------------------------
##----------------------------------Neural Network - Run Function-------------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------
    if Run_Function == 'Run Function':
        cnn = CNN_Convolution_Neural_Network.CNN()
        cnn.split_and_normalize_data(Dataset)
        cnn.create_neural_network()
        cnn.run_model(optimizer=optimizer, loss=loss, batch_size=batch_size, n_epochs=n_epochs)
        cnn.loss_function()
        cnn.accuracy()
        cnn.predict_test_data()
        cnn.confusion_matrix1()
        cnn.misclassified()
        # cnn.import_new_data('dress.JPG')
        # cnn.predict_new_data()
        cnn.visualize_training_data()
        # cnn.save_and_load_model()
        # cnn.visualize_model()

##-------------------------------------------main page------------------------------------------------------------------
##----------------------------------------tabs and Graphs---------------------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([ "loss function per epoch graph", "accuracy per epoch graph",
                                           "confusion_matrix", "Misclassified Sample", "visualize_training_data"])
        with tab1:
            st.header("Loss Function Per Epoch")
            NN_graph = Image.open('Loss_Function_CNN.png')
            st.image(NN_graph, caption='Loss_Function_CNN.png')
        with tab2:
            st.header("Accuracy Per Epoch")
            LossFunctionPerEpoch = Image.open('accuracy_CNN.png')
            st.image(LossFunctionPerEpoch, caption='accuracy_CNN.png')
        with tab3:
            st.header("Confusion Matrix")
            LossFunctionPerEpoch = Image.open('Confusion_matrix.png')
            st.image(LossFunctionPerEpoch, caption='Confusion_matrix.png')
        with tab4:
            st.header("Misclassified Sample")
            LossFunctionPerEpoch = Image.open('misclassified_index.png')
            st.image(LossFunctionPerEpoch, caption='misclassified_index.png')
        with tab5:
            st.header("Visualize Train Data")
            LossFunctionPerEpoch = Image.open('visualize_data_training.png')
            st.image(LossFunctionPerEpoch, caption='misclassified_index.png')
        with tab6:
            st.header("Python Code")
            st.code(open("CNN_Convolution_Neural_Network.py").read(), language="python")

    elif Run_Function == 'Dont Run Function':
        print('press run function')
    else:
        print('Error with running button')

##-----------------------------------------------------------------------------------------------------------
main()



