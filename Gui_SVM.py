import streamlit as st
import SVM
from sklearn import datasets
from PIL import Image

st.title('SVM')

col1, col2 = st.columns([2,3])

with col1:
    add_dataset = st.radio(
        'Which DataSet do you want to use?',
        ('iris', 'wine', 'brest cancer', 'diabetes'))

    add_test_size = st.slider('test_size parameter/100 in train_test_split() function(default 0.05).', 0, 100, 5)
    add_test_size = add_test_size / 100
    add_random_state = st.slider('random_state parameter in train_test_split() function(default 42).', 0, 100, 42)

    add_kernel = st.radio(
        'Which kernel in SVC function do you want to use? \n (raise AttributeError: coef_ is only available when using a linear kernel!!)',
        ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'))

    add_C = st.slider('C parameter in SVC function(default 1).', 1, 10, 1)

if add_dataset == 'iris' :
    dataset = datasets.load_iris()
elif add_dataset == 'diabetes':
    dataset = datasets.load_diabetes()
elif add_dataset == 'wine':
    dataset = datasets.load_wine()
elif add_dataset == 'brest cancer' :
    dataset = datasets.load_breast_cancer()
else:
    print('Error to input dataset')

S = SVM.svm(dataset, test_size=add_test_size, random_state=add_random_state, kernel=add_kernel, c=add_C)
S.print_data()

with col2:
    image = Image.open("SVM.png")
    st.image(image)
