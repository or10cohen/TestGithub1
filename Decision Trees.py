# from PIL import Image
from IPython.display import Image
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
from sklearn.tree import export_graphviz
from six import StringIO
import pydotplus
import graphviz
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset = True)


class DecisionTrees():

    def __init__(self):
        self.df, self.dfShape = self.ImportData()
        self.X_train, self.X_test, self.y_train, self.y_test, self.X, self.y = self.SplittDataSet()
        self.Gini = self.GiniIndex()
        self.dtree = self.TrainingData()
        self.predictions = self.TestAccuracy()
        self.ploting = self.Ploting()

    def __str__(self):
        pass
##-------------------------------------------------Import Data----------------------------------------------------------
    def ImportData(self):
        print(Fore.LIGHTYELLOW_EX + '\n---------------------------------Import Data---------------------------------')
        df = pd.read_csv('kyphosis.csv')            #import data
        print(Fore.RED + '\nfirst we can print the first 5 rows in data use command: df.head()\n', df.head())
        print(Fore.RED + '\nwe check the size, columns title, no. of row and more info data use command: df.info()')
        print(df.info())
        print(Fore.RED + '\nand we can use other commands like:'
                , Fore.BLUE +  'df.shape', df.shape
                , Fore.BLUE +  'df.size', df.size)
        ##sns.pairplot(df, hue='Kyphosis')
        return df, df.shape
##--------------------------------------Splitting the Dataset in Train\Test---------------------------------------------
    def SplittDataSet(self):
        print(Fore.LIGHTYELLOW_EX + '\n-------------------Splitting the Dataset to Train\Test-----------------------')
        X = self.df.drop('Kyphosis', axis=1)
        y = self.df['Kyphosis']
        print(Fore.RED + '\nFor X(data) we use the all data *without* the label, column \'Kyphosis\'. use the command:'
              , Fore.BLUE + 'X = df.drop(\'Kyphosis\', axis=1)  '
              , Fore.RED + '\nNOTE: axis=1 for drop column\n', self.df.drop('Kyphosis', axis=1)
              , Fore.RED + '\n\nFor y(label) we use just the \'Kyphosis\' column. use the command:'
              , Fore.BLUE + 'y = df[\'Kyphosis\']\n', self.df['Kyphosis'])
        # from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        print(Fore.RED + '\nSplit the data to Train/Test: in the first time we use sklearn library.\n'
                 'To split the data we use the library: from sklearn.model_selection import train_test_split'
                 , Fore.BLUE + '\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)')
        return X_train, X_test, y_train, y_test, X, y
##--------------------------------------------Gini index manually-------------------------------------------------------

    def GiniIndex(self, SplitValueAge = 36, SplitValueNumber = 5, SplitValueStart = 10):
        print(Fore.LIGHTYELLOW_EX + '--------------------------Gini & Entropy index manually------------------------------------')
        ListColumnNames = list(self.df.columns)
        ListColumnNames.pop(0)
        SplitValue = [SplitValueAge, SplitValueNumber, SplitValueStart]
        EntropyDictForAnyNodes = {str(ListColumnNames[0]) + 'Entropy': 0, str(ListColumnNames[1])
                                + 'Entropy': 0, str(ListColumnNames[2]) + 'Entropy': 0}
        EntropyValuesForEveryEdge = [[], [], []]
        for i in range(len(ListColumnNames)):
            PYesEdge1SmallerThen = self.X_train[ListColumnNames[i]] <= SplitValue[i]
            # st.dataframe(PYesEdge1SmallerThen)
            PYesEdge1SmallerThenAndEqual = self.y_train[PYesEdge1SmallerThen] == 'absent'
            PYesEdge1SmallerThenAndEqualAndTrue = PYesEdge1SmallerThenAndEqual[PYesEdge1SmallerThenAndEqual] == True
            PYesEdge1SmallerThenAndEqualAndFalse = len(PYesEdge1SmallerThenAndEqual) - len(PYesEdge1SmallerThenAndEqualAndTrue)
            LenPYesEdge1SmallerThenAndEqualAndTrue = len(PYesEdge1SmallerThenAndEqualAndTrue)
            LenPYesEdge1SmallerThenAndEqualAndFalse = PYesEdge1SmallerThenAndEqualAndFalse
            EntropyValuesForEveryEdge[i].append(LenPYesEdge1SmallerThenAndEqualAndTrue)
            EntropyValuesForEveryEdge[i].append(LenPYesEdge1SmallerThenAndEqualAndFalse)

            PYesEdge1BiggerThen = self.X_train[str(ListColumnNames[i])] > SplitValue[i]
            PYesEdge1BiggerThenAndEqual = self.y_train[PYesEdge1BiggerThen] == 'absent'
            PYesEdge1BiggerThenAndEqualAndTrue = PYesEdge1BiggerThenAndEqual[PYesEdge1BiggerThenAndEqual] == True
            PYesEdge1BiggerThenAndEqualAndFalse = len(PYesEdge1BiggerThenAndEqual) - len(PYesEdge1BiggerThenAndEqualAndTrue)
            LenPYesEdge1BiggerThenAndEqualAndTrue = len(PYesEdge1BiggerThenAndEqualAndTrue)
            LenPYesEdge1BiggerThenAndEqualAndFalse = PYesEdge1BiggerThenAndEqualAndFalse
            EntropyValuesForEveryEdge[i].append(LenPYesEdge1BiggerThenAndEqualAndTrue)
            EntropyValuesForEveryEdge[i].append(LenPYesEdge1BiggerThenAndEqualAndFalse)


            if sum(EntropyValuesForEveryEdge[i]) != len(self.X_train[str(ListColumnNames[i])]):
                print(EntropyValuesForEveryEdge[i])
                print(sum(EntropyValuesForEveryEdge[i]))
                raise ValueError('you need to use all options in edges to calculate the Gini index!!! ')


        EntropyEdges = []
        EntropyNodes = []
        IG = []
        for i in EntropyValuesForEveryEdge:
            p0, p1, p2, p3= i[0], i[1], i[2], i[3]
            if p0 == 0 or p1 == 0:
                EEdge0 = 0
                # raise ValueError('some of P are equal to zero!!')
            else:
                EEdge0 = - ((p0 / (p0 + p1)) * np.log2(p0 / (p0 + p1)) + (p1 / (p0 + p1)) * np.log2(p1 / (p0 + p1)))

            if p2 == 0 or p3 == 0:
                EEdge1 = 0
                # raise ValueError('some of P are equal to zero!!')
            else:
                EEdge1 = - ((p2 / (p2 + p3)) * np.log2(p2 / (p2 + p3)) + (p3 / (p2 + p3)) * np.log2(p3 / (p2 + p3)))
            EntropyEdges.append(EEdge0)
            EntropyEdges.append(EEdge1)

            EntropyNode = (p0 + p1) / sum(i) * EEdge0 + (p2 + p3) / sum(i) * EEdge1
            # print(EntropyNode)
            EntropyNodes.append(EntropyNode)

        EntropyEdges = [[EntropyEdges[0], EntropyEdges[1]], [EntropyEdges[2], EntropyEdges[3]],
                        [EntropyEdges[4], EntropyEdges[5]]]
        print('EntropyEdges:', EntropyEdges)
        EntropyDictForAnyNodes['AgeEntropy'] = EntropyNodes[0]
        EntropyDictForAnyNodes['NumberEntropy'] = EntropyNodes[1]
        EntropyDictForAnyNodes['StartEntropy'] = EntropyNodes[2]
        print('EntropyNode', EntropyNodes)
        print(EntropyDictForAnyNodes)

##---------------------------Training the Decision Tree Classifier------------------------------------------------------
    def TrainingData(self):
        print(Fore.LIGHTYELLOW_EX + '\n-----------------Training the Decision Tree Classifier-----------------------\n')
        # from sklearn.tree import DecisionTreeClassifier
        dtree = DecisionTreeClassifier(criterion="entropy")
        dtree.fit(self.X_train, self.y_train)
        print(Fore.RED + '\nWe have used the Gini index as our attribute selection method for the'
                 'training of decision tree classifier with Sklearn function:'
                 ,Fore.BLUE + 'DecisionTreeClassifier().')
        print(Fore.RED + '\nFinally, we do the training process by using the method: '
                 ,Fore.BLUE + 'model.fit()')

        return dtree
##--------------------------------------------Test Accuracy-------------------------------------------------------------
    def TestAccuracy(self):
        print(Fore.LIGHTYELLOW_EX + '\n----------------------------Test Accuracy------------------------------------\n')
        print(Fore.RED + '\nWe will now test accuracy by using the classifier on test data.'
                'For this we first use the dtree.predict function and pass X_test as attributes.\n'
                ,Fore.BLUE + 'predictions = dtree.predict(X_test)')
        predictions = self.dtree.predict(self.X_test)
        #from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        accuracy = accuracy_score(self.y_test, predictions)
        print(Fore.RED + '\n accuracy_score(y_test,predictions)\n', accuracy)
        # print(Fore.RED + '\n confusion_matrix(y_test, predictions)\n', confusion_matrix(self.y_test, predictions))
        # print(Fore.RED + '\nclassification_report(y_test, predictions)\n', classification_report(self.y_test, predictions))
        return predictions

    def Ploting(self):
        print(Fore.LIGHTYELLOW_EX + '\n----------------------------Ploting------------------------------------\n')
        clf_model = DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=3, min_samples_leaf=5)
        clf_model.fit(self.X_train, self.y_train)

        target = list(self.df['Kyphosis'].unique())
        feature_names = list(self.X.columns)
        # from sklearn import tree
        # import graphviz
        dot_data = tree.export_graphviz(clf_model,
                                        out_file=None,
                                        feature_names=feature_names,
                                        class_names=target,
                                        filled=True, rounded=True,
                                        special_characters=True)


TestAreTree = DecisionTrees()
