import streamlit as st
import SVM
from sklearn import datasets
from PIL import Image

st.title('SVM')

# Add a selectbox to the sidebar:
add_dataset = st.radio(
    'Which DataSet do you want to use?',
    ('iris', 'wine', 'brest cancer', 'diabetes'))

add_test_size = st.slider('test_size parameter in train_test_split() function', 0, 100, 5)
add_random_state = st.slider('random_state parameter in train_test_split() function', 0, 100, 42)

add_kernel = st.radio(
    'Which kernel in SVC function do you want to use?',
    ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'))

add_C = st.slider('C parameter in SVC function', 0, 2, 1)


if add_dataset == 'iris' :
    dataset = datasets.load_iris()
elif add_dataset == 'diabetes':
    dataset = datasets.load_diabetes()
elif add_dataset == 'wine':
    dataset = datasets.load_wine()
elif add_dataset == 'brest cancer' :
    dataset = datasets.load_breast_cancer()
else:
    pass

S = SVM.svm(dataset, test_size=add_test_size, random_state=add_random_state, kernel=add_kernel, c=add_C)

image = Image.open('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\SVM.png')
st.image(image)
