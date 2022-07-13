import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

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

    def __init__(self, df):
        self.df = df
    def __str__(self):
        print(Fore.RED + '\ndf.info()\n')
        print(self.df.info())
        print(Fore.RED + '\ndf.shape\n', self.df.shape)
        print(Fore.RED + '\ndf.head(5)\n', self.df.head(5))
    def preProccesing(self):
        ###-------------------------duplicate_rows-------------------------------#
        duplicate_rows = self.df[self.df.duplicated()]
        # print(Fore.RED + "\nNumber of duplicate rows:\n", duplicate_rows.shape[0])
        ###-------------------------Looking for null values-------------------------------#
        # print(Fore.RED + "\nNull values per column:\n", self.heart_disease.isnull().sum().to_frame('nulls'))
        limit_Null = self.df.shape[0] / 3
        nulls = self.df.isnull().sum().to_frame()
        # print(Fore.RED + '\nthe next columns have a lot of Null(bigger then len column {} / 3):'.format(
        #     self.heart_disease.shape[0]))
        column_with_big_null = []
        for index, row in nulls.iterrows():
            if row[0] > limit_Null:
                # print(index, row[0])
                column_with_big_null.append(str(index))
        if len(column_with_big_null) == 0:
            # print('we dont have columns with a lot of Null')
            pass
        else:
            self.df.drop(column_with_big_null, inplace=True, axis=1)
        # print(Fore.RED + '\nlets remove the all columns with large null(if we have)\n', self.heart_disease.head(5))
        ###---------------------------replace other Null with median---------------------------------------#
        # print(Fore.RED + "\nreplace other Null with average or other way:")
        column_fillna_median = ['trestbps', 'chol', 'thalch', 'oldpeak']
        # print(Fore.RED + "for {} we take the median instead of Null.\n".format(column_fillna_median))
        for column in column_fillna_median:
            median = self.df[column].median()
            self.df[column] = self.df[column].fillna(median)
        # print(Fore.RED + "\nNull values per column:\n", self.heart_disease.isnull().sum().to_frame('nulls'))
        ###-----------------------------what null are left???--------------------------------------------#
        # print(Fore.RED + 'we have some null are left so lets dig inside')
        # print(Fore.RED + 'unique_value per column')
        # print('fbs unique_value', self.heart_disease['fbs'].unique())
        # print('restecg unique_value', self.heart_disease['restecg'].unique())
        # print('exang unique_value', self.heart_disease['exang'].unique())
        # print(Fore.RED + 'lets delete the restecg Null rows, because we have just 2 \
        #              \nand for exang we delete too because we have three value inside')
        index_Null_restecg = self.df[self.df['restecg'].isnull()].index.tolist()
        index_Null_exang = self.df[self.df['exang'].isnull()].index.tolist()
        # print(index_Null_restecg)
        # print(index_Null_exang)
        self.df.drop(index_Null_restecg, inplace=True, axis=0)
        self.df.drop(index_Null_exang, inplace=True, axis=0)
        # print(Fore.RED + 'and for the last one, fbs, we put random value True/False')
        self.df['fbs'] = self.df['fbs'].fillna(bool(random.choice([True, False])))
        # print(Fore.RED + "\nNull values per column:\n", self.heart_disease.isnull().sum().to_frame('nulls'))
        self.df.drop(['id', 'dataset'], inplace=True, axis=1)
        # print(Fore.RED + 'delete id & dataset columns and convert True False columns to 1/0 with Dummy\n',
        #       self.heart_disease.head(5))
        self.df = pd.get_dummies(self.df, columns=['sex', 'exang', 'fbs'], drop_first=True)
        # print(Fore.RED + 'continue to other feature with more then 2 value. pay attention its create more then 1 new column\n',
        #       self.heart_disease.head(5))
        self.df = pd.get_dummies(self.df, columns = ['restecg', 'cp'])
        # print(Fore.RED + 'delete id & dataset columns\n', self.heart_disease.head(5))
        last_column = self.df.pop('num')
        self.df.insert(15, 'num', last_column)
        print(Fore.RED + 'after preprocessing\n', self.df.head(5))
    def correlation(self):
        pearsonCorr = self.df.corr(method='pearson')
        fig = plt.subplots(figsize=(14, 8))
        maskP = np.triu(np.ones_like(pearsonCorr, dtype=bool))
        maskP = maskP[1:, :-1]
        pCorr = pearsonCorr.iloc[1:, :-1].copy()
        cmap = sns.diverging_palette(0, 200, 150, 50, as_cmap=True)
        sns.heatmap(pCorr, vmin=-1,vmax=1, cmap = cmap, annot=True, linewidth=0.3, mask=maskP)
        plt.title("Pearson Correlation")
        plt.savefig('correlation.png')
        self.df.drop(['thalch', 'fbs_True', 'restecg_normal', 'restecg_st-t abnormality', 'cp_typical angina'], inplace=True, axis=1)
        print(Fore.RED + 'after correlation\n', self.df.head(5))


    def descionTree(self):
        X = self.df.iloc[:, :-1]  # Features
        y = self.df.iloc[:, -1]  # Target variable
        X = MinMaxScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                            random_state=1)  # 70% training and 30% test
        clf = DecisionTreeClassifier()
        self.clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(Fore.RED + "Accuracy:\n", metrics.accuracy_score(y_test, y_pred))
    def plotDescionTree(self):
        columns = []
        for col in self.df.columns:
            columns.append(str(col))
        columns.pop()
        print(columns)
        dot_data = StringIO()
        export_graphviz(self.clf, out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True, feature_names=columns, class_names=['0', '1', '2'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png('diabetes.png')
        Image(graph.create_png())



if __name__ == '__main__':
    RunAdaBoost = AdaBoost(heart_disease)
    RunAdaBoost.__str__()
    RunAdaBoost.preProccesing()
    RunAdaBoost.correlation()
    RunAdaBoost.descionTree()
    # RunAdaBoost.plotDescionTree()


