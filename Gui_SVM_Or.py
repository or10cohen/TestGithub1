import streamlit as st
import SVM_Or
from sklearn import datasets
from PIL import Image


def main():
    st.title('SVM')
    col1, col2, col3 = st.columns([2, 4, 1])
    with col1:
        add_dataset = st.radio(
            'Which DataSet do you want to use?',
            ('brest cancer', 'iris', 'wine', 'diabetes'))

        add_test_size = st.slider('test_size parameter/100 in train_test_split() function(default 0.05).', 0, 100, 5)
        add_test_size = add_test_size / 100
        add_random_state = st.slider('random_state parameter in train_test_split() function(default 42).', 0, 100, 42)
        add_kernel = st.radio(
            'Which kernel in SVC function do you want to use? \n (raise AttributeError: coef_ is only available when using a linear kernel!!)',
            ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'))

        add_C = st.slider('C parameter in SVC function(default 1).', 1, 10, 1)

    with col3:
        add_table_parameter_min_x = st.slider('table_parameter_min_x', -100, 100, 0)
        add_table_parameter_max_x = st.slider('table_parameter_max_x', -100, 100, 0)
        add_table_parameter_min_y = st.slider('table_parameter_min_y', -100, 100, 0)
        add_table_parameter_max_y = st.slider('table_parameter_max_y', -100, 100, 0)


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

    S = SVM_Or.svm(dataset, test_size=add_test_size, random_state=add_random_state, kernel=add_kernel, c=add_C)



    S.plot_2d(table_parameter_min_x=add_table_parameter_min_x/100, table_parameter_max_x=add_table_parameter_max_x/100, \
              table_parameter_min_y=add_table_parameter_min_y/100, table_parameter_max_y=add_table_parameter_max_y/100)

    with col2:
        image = Image.open("SVM_Or.png")
        st.image(image)

if __name__ == "__main__":
    main()