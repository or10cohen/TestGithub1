import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import Neural_Network_TF_Regression_Code
import Neural_Network_TF_Classification_Code
from PIL import Image
# from dash import html
# import dash_bootstrap_components as dbc


def main():
    global Run_Function
    st.title('Neural Network - by Oren Niazov')

    if st.button('Run Function'):
        st.write('Running Function')
        Run_Function = 'Run Function'
    else:
        st.write('Press run function to start')
        Run_Function = 'Dont Run Function'
##---------------------------------------------------sidebar------------------------------------------------------------
    with st.sidebar:
        st.sidebar.title('Input Parameters')
        st.title('parameters for function: create_neural_network')
        No_hidden_layers = st.number_input('No_hidden_layer', value=3)
        st.write(No_hidden_layers)


        No_neurons_per_layer = st.text_input('No_neurons_per_layers = [int list (default=4)]: \n '
                                             'input example 3 hidden layers : 444 \n', value='None')
        if No_neurons_per_layer != 'None':
            No_neurons_per_layer = [int(i) for i in str(No_neurons_per_layer)]
        else:
            pass
        st.write('No_neurons_per_layer:', No_neurons_per_layer)

        activation_per_layer = st.text_input('activation_per_layer = [string list (default=\'relu\')]: \n '
                                             'input example 3 hidden layers: relu relu relu\n'
                                             'but you can use: sigmoid, tanh and more', value='None')
        if activation_per_layer != 'None':
            def Convert(string):
                li = list(string.split(" "))
                return li
            activation_per_layer = Convert(activation_per_layer)
        else:
            pass
        st.write('No_neurons_per_layer:', activation_per_layer)







        st.title('parameters for function: run_model')
        optimizer = st.selectbox(
            'choose optimizer',
            ('SGD','rmsprop', 'adam', 'adamax', 'Nadam'))
        st.write('optimizer', optimizer)
        loss = st.selectbox(
            'choose loss function',
            ('mse', 'mse', 'mse'))
        st.write('loss', loss)
        n_epoch = st.number_input('random_state:', value=250)
        st.write('n_epochs:', n_epoch)


        st.title('choose Datatype&Dataset')
        Datatype = st.selectbox(
            'choose Datatype',
            ('Regression', 'Classification'))
        st.write('Datatype:', Datatype)

        if Datatype=='Regression':
            Dataset = st.selectbox(
                'choose Dataset',
                ('fake_regression0', 'fake_regression1', 'fake_regression2'))
            st.write('Dataset:', Dataset)

        elif Datatype=='Classification':
            Dataset = st.selectbox(
                'choose Dataset',
                ('fake_classification0', 'classification1', 'fake_classification2'))
            st.write('Dataset:', Dataset)
        else:
            print('pay attention: something wrong with the Datatype choose')


        st.title('parameters for function: split_and_normalize_data')
        test_size = st.number_input('test_size', value=0.3, format="%.2f")
        st.write('test_size:', test_size)
        random_state = st.number_input('random_state:', value=42)
        st.write('random_state:', random_state)


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


        ##----------------------download files you want to share-----------------------
        with open("Neural_Network_TF_Classification_Code.py") as file:
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

    if Run_Function == 'Run Function':
        run = Neural_Network_TF_Regression_Code.FirsRegressionNeuralNetwork(X, y, n_epochs=n_epoch)
        run.split_and_normalize_data(test_size=test_size, random_state=random_state)
        run.create_neural_network(No_hidden_layers=No_hidden_layers, No_neurons_per_layer=No_neurons_per_layer,
                              activation_per_layer=activation_per_layer)
        run.run_model()
        run.epochs_graph()
        run.predict()
        # run.save_and_load_model()
    elif Run_Function == 'Dont Run Function':
        print('press run function')

    NN_graph = Image.open('NN_graph.png')
    st.image(NN_graph, caption='NN_graph.png')
    LossFunctionPerEpoch = Image.open('Graph.png')
    st.image(LossFunctionPerEpoch, caption='Loss Function Per Epoch')
##-----------------------------------------------------------------------------------------------------------
    # Grpah = st.radio(
    #     "Choose graph",
    #     ('Tx and Rx RMS current Load', 'Tx and Rx RMS current Min Load', 'arona - Efficency'))
    #
    # if Grpah == 'Tx and Rx RMS current Load':
    #     st.image(Varona0, caption='Sunrise by the mountains')
    # elif Grpah == 'Tx and Rx RMS current Min Load':
    #     st.image(Varona1, caption='Sunrise by the mountains')
    # elif Grpah == 'arona - Efficency':
    #     st.image(Varona2, caption='Sunrise by the mountains')
    # else:
    #     st.write("You didn't select comedy.")

##-----------------------------------------------------------------------------------------------------------
    # carousel = dbc.Carousel(
    #     items=[
    #         {"key": "1", "src": "Varona - Vrect loaded (200W) vs. Min load - FB DC = 20 present.png"},
    #         {"key": "2", "src": "Varona - Efficency.png"},
    #         {"key": "3", "src": "Varona - Tx and Rx RMS current Min Load.png"},
    #         {"key": "3", "src": "Varona - Tx and Rx RMS current Min Load.png"},
    #     ],
    #     controls=True,
    #     indicators=True,
    # )
##-----------------------------------------------------------------------------------------------------------
main()


