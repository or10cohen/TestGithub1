import streamlit as st
import Hiercrchical_Clustering
from sklearn import datasets
from PIL import Image

st.title('Hiercrchical Clustering')


col1, col2 = st.columns([1,2])
add_cluster=2
max_distance=9999

with col1:
    add_selectbox = st.radio(
        'Which DataSet you want to use?',
        ('iris', 'wine', 'breast cancer', 'diabetes'))
    add_linkage = st.radio(
        'Which Linkage you want to use?',
        ('complete', 'average', 'single'))

    add_Hyperparameter = st.radio(
        'Which Hyperparameter you want?',
        ('Cluster', 'Max distance'))

    if add_Hyperparameter == 'Cluster':
        add_cluster = st.radio(
            'How many cluster you want?',
            (2, 3, 4))
    elif add_Hyperparameter == 'Max distance':
        max_distance = st.slider('Max distance / 100', 0, 200, 200, step=1)
    else:
        pass
    add_dimensions = st.radio(
        'Plot with how many dimensions?',
        (2, 3))

    if add_dimensions == 3:
        rotate_fig_0 = st.slider('Rotate axis x', 0, 180, 45, step=45)
        rotate_fig_1 = st.slider('Rotate axis y', 0, 180, 45, step=45)
    else:
        pass


if add_selectbox == 'iris' :
    dataset = datasets.load_iris()
elif add_selectbox == 'diabetes':
    dataset = datasets.load_diabetes()
elif add_selectbox == 'wine':
    dataset = datasets.load_wine()
elif add_selectbox == 'breast cancer' :
    dataset = datasets.load_breast_cancer()
else:
    pass
X = dataset.data[:, :]

HC = Hiercrchical_Clustering.HierarchicalClustering(X, number_clusters=add_cluster, max_distance=max_distance / 100, linkage_method=add_linkage)
HC.fit()

with col2:
    if add_dimensions == 2:
        HC.print_2d()
        image = Image.open('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\Print_2d.png')
        st.image(image)
    elif add_dimensions == 3:
        HC.print_3d(rotate_fig_0=rotate_fig_0, rotate_fig_1=rotate_fig_1)
        image = Image.open('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\Print_3d.png')
        st.image(image)
    else:
        pass




#  py -m streamlit run filename.py
# st run C:/Users/or_cohen/PycharmProjects/TestGithub1/Gui_Hiercrchical_Clustering.py [ARGUMENTS]