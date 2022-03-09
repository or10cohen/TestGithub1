import streamlit as st
import Hiercrchical_Clustering
from sklearn import datasets
from PIL import Image

st.title('Hiercrchical Clustering')


# Add a selectbox to the sidebar:
add_selectbox = st.radio(
    'Which DataSet you want to use?',
    ('iris', 'wine', 'brest cancer', 'diabetes'))
add_linkage = st.radio(
    'Which Linkage you want to use?',
    ('complete', 'average', 'single'))
add_cluster = st.radio(
    'How many cluster you want(Hyperparameter)?',
    (2, 3, 4))
add_dimensions = st.radio(
    'Plot with how many dimensions?',
    (2, 3))

if add_dimensions == 3:
    rotate_fig_0 = st.slider('How old are you?', 0, 180, 25)
    rotate_fig_1 = st.slider('How dddold are you?', 0, 180, 25)
else:
    pass


if add_selectbox == 'iris' :
    dataset = datasets.load_iris()
elif add_selectbox == 'diabetes':
    dataset = datasets.load_diabetes()
elif add_selectbox == 'wine':
    dataset = datasets.load_wine()
elif add_selectbox == 'brest cancer' :
    dataset = datasets.load_breast_cancer()
else:
    pass
X = dataset.data[:, :]


HC = Hiercrchical_Clustering.HierarchicalClustering(X, number_clusters=add_cluster, linkage_method=add_linkage)
HC.fit()
if add_dimensions == 2 :
    HC.Print_2d()
    image = Image.open('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\Print_2d.png')
    st.image(image)
elif add_dimensions == 3:
    HC.Print_3d(rotate_fig_0=rotate_fig_0, rotate_fig_1=rotate_fig_1)
    image = Image.open('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\Print_3d.png')
    st.image(image)
else:
    pass


# run_my_function = Hiercrchical_Clustering.HierarchicalClustering(X)

# st run C:/Users/or_cohen/PycharmProjects/TestGithub1/Gui.py [ARGUMENTS]