import streamlit as st
import DBSCAN
from sklearn import datasets
from PIL import Image



def main():
    st.title('DBSCAN')
    col1, col2 = st.columns([1,2])
    with col1:
        add_dataset = st.radio(
            'Which DataSet do you want to use?',
            ('iris', 'brest cancer', 'wine', 'diabetes'))

        epsilon = st.slider('epsilon parameter/100 (default 0.3).', 0, 100, 30)
        epsilon = epsilon / 100
        minPts = st.slider('minPts for corePts (default 3)', 0, 10, 3)
        add_dimensions = st.radio(
            'Plot with how many dimensions?',
            (2, 3))

        if add_dimensions == 3:
            rotate_fig_0 = st.slider('Rotate axis x', 0, 180, 45, step=45)
            rotate_fig_1 = st.slider('Rotate axis y', 0, 180, 45, step=45)
        else:
            pass


    if add_dataset == 'brest cancer' :
        dataset = datasets.load_breast_cancer()
    elif add_dataset == 'iris' :
        dataset = datasets.load_iris()
    elif add_dataset == 'diabetes':
        dataset = datasets.load_diabetes()
    elif add_dataset == 'wine':
        dataset = datasets.load_wine()
    else:
        print('Error to input dataset')

    run_DBSCAN = DBSCAN.DBSCAN(dataset, number_of_features=add_dimensions, epsilon=epsilon, minPts=minPts)
    run_DBSCAN.run()

    with col2:
        if add_dimensions == 2:
            run_DBSCAN.print_2d()
            image = Image.open('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\DBSACN_2D.png')
            st.image(image)
        elif add_dimensions == 3:
            run_DBSCAN.print_3d(rotate_fig_0=rotate_fig_0, rotate_fig_1=rotate_fig_1)
            image = Image.open('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\DBSACN_3D.png')
            st.image(image)
        else:
            pass

if __name__ == "__main__":
    epsilon = 0.3
    minPts = 3
    ataset = datasets.load_iris()
    main()