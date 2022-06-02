import streamlit as st
import DBSCAN
from sklearn import datasets
from PIL import Image
from sklearn.utils import Bunch

epsilon = 0.3
minPts = 3
dataset = datasets.load_iris()


def main():
    st.title('DBSCAN')
    col1, col2, col3 = st.columns([2,5,2])
    with col1:
        add_dataset = st.radio(
            'Which DataSet do you want to use?',
            ('iris', 'circle in circle 2D', 'wine', 'diabetes'))

        epsilon = st.slider('epsilon parameter/100 (default 0.3).', 0, 100, 30)
        epsilon = epsilon / 100
        minPts = st.slider('minPts for corePts (default 3)', 0, 10, 3)
        number_of_features = st.slider('how many features so you want to use?', 0, 30, 2)
        add_dimensions = st.radio(
            'Plot with how many dimensions?',
            (2, 3))


    if add_dataset == 'circle in circle 2D':
        dataset = datasets.make_circles(n_samples=100, shuffle=True, noise=None, random_state=None, factor=0.4)
        dataset = Bunch(data=dataset[0])
    elif add_dataset == 'iris' :
        dataset = datasets.load_iris()
    elif add_dataset == 'diabetes':
        dataset = datasets.load_diabetes()
    elif add_dataset == 'wine':
        dataset = datasets.load_wine()
    else:
        print('Error to input dataset')


    run_DBSCAN = DBSCAN.DBSCAN(dataset, number_of_features=number_of_features, epsilon=epsilon, minPts=minPts)
    run_DBSCAN.run()

    with col3:
        if add_dimensions == 3:
            rotate_fig_0 = st.slider('Rotate axis x', 0, 180, 45, step=45)
            rotate_fig_1 = st.slider('Rotate axis y', 0, 180, 45, step=45)
        else:
            pass

    with col2:
        if add_dimensions == 2:
            run_DBSCAN.plot_2d(run_DBSCAN.cluster)
            image = Image.open('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\DBSACN_2D.png')
            st.image(image)

        elif add_dimensions == 3:
            if number_of_features <= 2:
                st.markdown(f'<h1 style="color:#990000;font-size:22px;">{"cant plot 3D with less then 3 feathers"}</h1>',
                            unsafe_allow_html=True)
            else:
                run_DBSCAN.plot_3d(run_DBSCAN.cluster, rotate_fig_0=rotate_fig_0, rotate_fig_1=rotate_fig_1)
                image = Image.open('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\DBSACN_3D.png')
                st.image(image)
        else:
            pass





main()
