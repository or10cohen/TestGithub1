import numpy as np
import pandas
import pandas as pd
import seaborn as sns
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
        print(Fore.RED + '\nwe have a lot of Null in columns:')
        column_with_big_null = []
        for index, row in nulls.iterrows():
            if row[0] > limit_Null:
                print(index, row[0])
                column_with_big_null.append(str(index))
        heart_disease.drop(column_with_big_null, inplace=True, axis=1)
        print(Fore.RED + '\nlets remove the columns with large null\n', heart_disease.head(5))
        #---------------------------------------------------------------------------------------------#
        print(sns.boxplot(x=heart_disease['age']))



        # print(Fore.RED + '\nheart_disease.head(10) after dummies\n', heart_disease_copy.head(10))
        # print(Fore.RED + '\ncolumns\n', heart_disease_copy.columns)
        # print(Fore.RED + '\ndescribe\n', heart_disease_copy.describe())



if __name__ == '__main__':
    RunAdaBoost = AdaBoost()
    RunAdaBoost.__str__()


