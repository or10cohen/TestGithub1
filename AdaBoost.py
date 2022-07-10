import numpy as np
import pandas
import pandas as pd
# import seaborn as sns
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset = True)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

heart_disease = pd.read_csv('heart_disease_uci.csv')
heart_disease_copy = heart_disease

# dummies = pd.get_dummies(heart_disease_copy[['sex', 'fbs', 'restecg', 'exang']], drop_first=True)
# heart_disease_copy = pd.concat([heart_disease_copy.drop(['sex', 'fbs', 'restecg', 'exang'],axis=1), dummies],axis=1)
# heart_disease_copy.drop(['id', 'cp', 'dataset', 'slope', 'thal', 'restecg_st-t abnormality'], inplace=True, axis=1)

class AdaBoost():

    def __init__(self):
        pass

    def __str__(self):
        print(Fore.RED + '\nheart_disease.info()\n')
        print(heart_disease.info())
        print(Fore.RED + '\nheart_disease.shape\n', heart_disease.shape)
        print(Fore.RED + '\nheart_disease.head(10)\n', heart_disease.head(5))
        #-------------------------duplicate_rows-------------------------------#
        duplicate_rows = heart_disease[heart_disease.duplicated()]
        print(Fore.RED + "\nNumber of duplicate rows:\n", duplicate_rows.shape[0])
        #-------------------------Looking for null values-------------------------------#
        print(Fore.RED + "\nNull values per column:\n", heart_disease.isnull().sum().to_frame('nulls'))
        limit_Null = heart_disease.shape[0] / 3
        nulls = heart_disease.isnull().sum().to_frame()
        print(Fore.RED + '\nthe next columns have a lot of Null(bigger then len column {} / 3):' .format(heart_disease.shape[0]))
        column_with_big_null = []
        for index, row in nulls.iterrows():
            if row[0] > limit_Null:
                print(index, row[0])
                column_with_big_null.append(str(index))
        if len(column_with_big_null) == 0:
            print('we dont have columns with a lot of Null')
        else:
            heart_disease.drop(column_with_big_null, inplace=True, axis=1)
        print(Fore.RED + '\nlets remove the all columns with large null(if we have)\n', heart_disease.head(5))
        #---------------------------replace other Null with median---------------------------------------#
        print(Fore.RED + "\nreplace other Null with average or other way:")
        column_fillna_median = ['trestbps', 'chol', 'thalch', 'oldpeak']
        print(Fore.RED + "for {} we take the median instead of Null.\n" .format(column_fillna_median))
        for column in column_fillna_median:
            median = heart_disease[column].median()
            heart_disease[column] = heart_disease[column].fillna(median)
        print(Fore.RED + "\nNull values per column:\n", heart_disease.isnull().sum().to_frame('nulls'))
        # ------------------------------------------------------------------------------------------#


if __name__ == '__main__':
    RunAdaBoost = AdaBoost()
    RunAdaBoost.__str__()


