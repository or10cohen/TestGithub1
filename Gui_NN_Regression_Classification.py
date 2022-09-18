import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import Neural_Network_TF_Regression_Classification_Code
import Convolution_Neural_Network_TF
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
##----------------------------------Neural Network - by Oren Niazov-----------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------
    global Run_Function, Dataset
    st.title('**Neural Network - by Oren Niazov & Or Yosef Cohen**')

    if st.button('Run Function'):
        st.write('Running Function')
        Run_Function = 'Run Function'
    else:
        st.write('Press run function to start')
        Run_Function = 'Dont Run Function'
##---------------------------------------------------sidebar------------------------------------------------------------
##-------------------------------------------choose Datatype&Dataset----------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------
    with st.sidebar:
        st.sidebar.title('_Input Parameters_')
        st.title('choose Datatype&Dataset')
        Datatype = st.selectbox(
            'choose Datatype',
            ('Regression', 'Classification'))

        if Datatype=='Regression':
            Dataset = st.selectbox(
                'choose Dataset',
                ('fake_regression0', 'fake_regression1', 'fake_regression2'))

        elif Datatype=='Classification':
            Dataset = st.selectbox(
                'choose Dataset',
                ('fake_classification0', 'classification1', 'fake_classification2'))
            st.write('Dataset:', Dataset)
        else:
            print('pay attention: something wrong with the Datatype choose')


        if Dataset=='fake_regression0':
            df = pd.read_csv('fake_reg.csv')
            X = df[['feature1', 'feature2']].values
            y = df['price'].values
        elif Dataset=='fake_regression1':
            Dataset = None
        elif Dataset=='fake_regression2':
            Dataset = None
        elif Dataset=='fake_classification0':
            Dataset = None
        elif Dataset == 'fake_classification1':
            Dataset = None
        elif Dataset == 'fake_classification2':
            Dataset = None
        else:
            print('pay attention: something wrong with the Dataset choose')

##----------------------------------------------sidebar-----------------------------------------------------------------
##----------------------------------------create_neural_network---------------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------
        st.title('parameters for function: create_neural_network')
        No_hidden_layers = st.number_input('No. of hidden layers', value=3)
        #st.write(No_hidden_layers)

        No_neurons_per_layer = st.text_input('No. of neurons per layer', value='None')
        st.write('example 3 hidden layers : 4 4 4')
        if No_neurons_per_layer != 'None':
            No_neurons_per_layer = [int(i) for i in str(No_neurons_per_layer.replace(" ",""))]
        else:
            pass
            #print('Error in \'input No. of neurons per layer\'')
        st.write(No_neurons_per_layer)
        activation_per_layer = st.text_input('activation per layer \n', value='None')
        st.write('example 3 hidden layers: relu relu relu. \nbut you can use: sigmoid, tanh and more')
        if activation_per_layer != 'None':
            def Convert(string):
                li = list(string.split(" "))
                return li
            activation_per_layer = Convert(" ".join(activation_per_layer.split()))
        else:
            pass
            # print('Error in \'activation per layer\'')
        st.write(activation_per_layer)

        label_kind = st.selectbox(
            'which label your Data use?',
            ('regression', 'classification', 'multi classification'))
        #st.write('label_kind', label_kind)

        if label_kind == 'classification':
            No_output_neurons = int(1)
        elif label_kind == 'multi classification':
            No_output_neurons = int(len(np.unique(y)))
        elif label_kind == 'regression':
            No_output_neurons = int(1)
        else:
            print('Or')
        st.write('Number of output neurons:', No_output_neurons)
##------------------------------------------------sidebar---------------------------------------------------------------
##----------------------------------------------- run_model-------------------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------
        st.title('parameters for function: run_model')
        optimizer = st.selectbox(
            'choose optimizer',
            ('rmsprop','SGD', 'adam', 'adamax', 'Nadam'))

        loss = st.selectbox(
            'choose loss function',
            ('MeanSquaredError[reg]', 'MeanAbsoluteError[reg]', 'MeanAbsolutePercentageError[reg]','MeanSquaredLogarithmicError[reg]',
            'BinaryCrossentropy[class]', 'CategoricalCrossentropy[class]', 'SparseCategoricalCrossentropy[class]', 'Poisson[class]'))

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
        n_epoch = st.number_input('No of epoch:', value=100)
        #st.write('n_epochs:', n_epoch)
        batch_size = st.number_input('batch_size:', value=32)
        # st.write('batch_size:', batch_size)

##----------------------------------------------------sidebar-----------------------------------------------------------
##------------------------------------------- split_and_normalize_data--------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------
        st.title('parameters for function: split_and_normalize_data')
        test_size = st.number_input('test_size', value=0.3, format="%.2f")
        st.write('test_size:', test_size)
        random_state = st.number_input('random_state:', value=42)
        st.write('random_state:', random_state)
##---------------------------------------------------sidebar------------------------------------------------------------
##-----------------------------------------download files you want to share---------------------------------------------
##----------------------------------------------------------------------------------------------------------------------
        with open("Convolution_Neural_Network_TF.py") as file:
            btn = st.download_button(
                label="Download Neural_Network_TF_Classification_Code Python Resources File",
                data=file,
            )
        with open("Neural_Network_TF_Regression_Code.py") as file:
            btn = st.download_button(
                label="Download Neural_Network_TF_Regression_Code Python Resources File",
                data=file,
            )
        with open("Gui_NN_Regression&Classification.py") as file:
            btn = st.download_button(
                label="Download Gui_NN_Regression&Classification Python Resources File",
                data=file,
            )

##----------------------------------------------------------------------------------------------------------------------
##----------------------------------------------main page---------------------------------------------------------------
##----------------------------------Neural Network - Run Function-------------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------
    if Run_Function == 'Run Function':
        run = Neural_Network_TF_Regression&Classification_Code.FirsRegressionNeuralNetwork(X, y)
        run.split_and_normalize_data(test_size=test_size, random_state=random_state)
        run.create_neural_network(No_hidden_layers=No_hidden_layers, No_neurons_per_layer=No_neurons_per_layer,
                                  activation_per_layer=activation_per_layer, No_output_neurons=No_output_neurons)
        run.run_model(optimizer=optimizer, loss=loss, batch_size=batch_size, n_epochs=n_epoch)
        run.epochs_graph()
        predict_test, check_new_data = run.predict()
        # run.save_and_load_model()

##-------------------------------------------main page------------------------------------------------------------------
##----------------------------------------tabs and Graphs---------------------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------

        tab1, tab2, tab3, tab4 = st.tabs([ "NN graph", "Loss Function Per Epoch", "Predict Table", "Python Code"])
        with tab1:
            st.header("NN graph")
            NN_graph = Image.open('NN_graph.png')
            st.image(NN_graph, caption='NN_graph.png')

        with tab2:
            st.header("Loss Function Per Epoch")
            LossFunctionPerEpoch = Image.open('Graph.png')
            st.image(LossFunctionPerEpoch, caption='Loss Function Per Epoch')

        with tab3:
            st.header("Predict Table")
            st.dataframe(data=predict_test, width=None, height=None)

        with tab4:
            st.header("Python Code")
            st.code(open("Neural_Network_TF_Regression_Code.py").read(), language="python")

    elif Run_Function == 'Dont Run Function':
        print('press run function')
    else:
        print('Error with running button')


##-----------------------------------------------------------------------------------------------------------
main()



