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
    epsilon = 0.000001
    st.title('Varona - by Oren Niazov')
    with st.sidebar:

        Datatype = st.selectbox(
            'choose Data type',
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


        st.title('parameters for: split_and_normalize_data def')
        test_size = st.number_input('test_size', value=0.3, format="%.10f")
        st.write('test_size:', test_size)
        random_state = st.number_input('random_state:', value=42, format="%.10f")
        st.write('random_state:', random_state)


        st.title('parameters for: create_neural_network def')
        No_hidden_layers = st.number_input('No_hidden_layer', value=3)
        st.write('No_hidden_layers:', No_hidden_layers)
        No_neurons_per_layer = st.text_input('No_neurons_per_layers = [int list (default=4)]:', value='None')
        st.write('No_neurons_per_layer:', No_neurons_per_layer)
        activation_per_layer = st.text_input('activation_per_layer = [string list (default=string(relu))]:', value='None')
        st.write('No_neurons_per_layer:', activation_per_layer)


        st.title('parameters for: run_model def')
        optimizer = st.selectbox(
            'choose optimizer',
            ('rmsprop', 'rmsprop', 'rmsprop'))
        st.write('optimizer', optimizer)
        loss = st.selectbox(
            'choose loss function',
            ('mse', 'mse', 'mse'))
        st.write('loss', loss)


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
        with open("CalcVrect.py") as file:
            btn = st.download_button(
                label="Download CalcVrect Python Resources File",
                data=file,
            )
        with open("Varona_k_range.py") as file:
            btn = st.download_button(
                label="Download Varona_k_range Python Resources File",
                data=file,
            )
        with open("Gui_Varona.py") as file:
            btn = st.download_button(
                label="Download Gui_Varona Python Resources File",
                data=file,
            )

    run = Neural_Network_TF_Regression_Code.FirsRegressionNeuralNetwork(X, y)
    run.split_and_normalize_data()
    run.create_neural_network()
    run.run_model()
    run.epochs_graph()
    run.predict()
    run.save_and_load_model()

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