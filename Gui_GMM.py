import streamlit as st
import numpy as np
import GMM
from sklearn import datasets
from PIL import Image
from sklearn.utils import Bunch
import pandas as pd

def main():
    st.title('GMM')
    col1, col2, col3 = st.columns([2,5,2])
    with col1:
        add_dataset = st.radio(
            'Which DataSet do you want to use?',
            ('make 3 circles 2D',  'iris', 'circle in circle 2D', 'blobs 2D', 's_curve 2D'))

        if add_dataset == 'circle in circle 2D':
            dataset = datasets.make_circles(n_samples=200, shuffle=True, noise=None, random_state=None, factor=0.4)
            dataset = Bunch(data=dataset[0])
        elif add_dataset == 'iris' :
            dataset = datasets.load_iris()
        elif add_dataset == 'make 3 circles 2D' :
            X_small, y_small = datasets.make_circles(n_samples=(200, 200), random_state=None,
                                                     noise=None, factor=0.1)
            X_large, y_large = datasets.make_circles(n_samples=(200, 200), random_state=None,
                                                     noise=None, factor=0.6)
            c = np.vstack((X_small,X_large))
            # c = np.stack((c, y_large))
            dataset = Bunch(data=c)
        elif add_dataset == 'blobs 2D':
            dataset = datasets.make_moons(n_samples=300, noise=0.05)
            dataset = Bunch(data=dataset[0])

        elif add_dataset == 's_curve 2D':
            centers = [(3, 3), (0, 0), (5, 5)]
            dataset = datasets.make_blobs(n_samples=200, centers=centers, shuffle=False, random_state=None)
            dataset = Bunch(data=dataset[0])
        else:
            print('Error to input dataset')
        if ('2D' in add_dataset):
            number_of_features = 2
            add_dimensions = 2
        else:
            number_of_features = st.slider('how many features so you want to use?', 0, 30, 2)
            add_dimensions = st.radio(
                'Plot with how many dimensions?',
                (2, 3))

        number_of_clusters = st.slider('number_of_clusters (default 2).', 1, 10, 2)
        n_epochs = st.slider('n_epochs (default 10)', 0, 100, 10)
        epsilon_likelihood = st.slider('epsilon_likelihood  (default 1)', 1, 100, 1)
        # epsilon_likelihood /= 10

    run_GMM = GMM.GMM(dataset.data, number_of_clusters=number_of_clusters, n_epochs=n_epochs, epsilon_likelihood=epsilon_likelihood)
    run_GMM.train_gmm()
    run_GMM.plot_likelihood()

    with col3:
        with open("GMM.py") as file:
            btn = st.download_button(
                label="Download Python Resources File",
                data=file,
            )

        if add_dimensions == 3:
            rotate_fig_0 = st.slider('Rotate axis x', 0, 180, 45, step=45)
            rotate_fig_1 = st.slider('Rotate axis y', 0, 180, 45, step=45)
        else:
            pass

    with col2:
        if add_dimensions == 2:
            run_GMM.train_gmm()
            image = Image.open('plot_likelihood.png')
            st.image(image)
            # htp = "https://raw.githubusercontent.com/djswoosh/Music-Recommendation-Engine-using-FMA-Dataset/main/1200px-The_Echo_Nest_logo.svg.png"
            # st.image(htp, caption='logo', width=350)
        elif add_dimensions == 3:
            run_GMM.train_gmm()
            image = Image.open('plot_likelihood.png')
            st.image(image)
        else:
            pass





main()
