import numpy
import numpy as np
import pandas as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset = True)


class DecisionTrees():

    def __init__(self):
        self.df, self.dfShape = self.ImportData()
        self.X_train, self.X_test, self.y_train, self.y_test = self.SplittDataSet()
        self.Gini = self.GiniIndex()
        self.dtree = self.TrainingData()
        self.predictions = self.TestAccuracy()

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
        return X_train, X_test, y_train, y_test
##--------------------------------------------Gini index manually-------------------------------------------------------

    def GiniIndex(self, SplitValueAge = 36, SplitValueNumber = 5, SplitValueStart = 10):
        print(Fore.LIGHTYELLOW_EX + '--------------------------Gini index manually------------------------------------')
        data = ['Age', 'Number', 'Start']
        SplitValue = [SplitValueAge, SplitValueNumber, SplitValueStart]
        GiniDict = {str(data[0]) + 'Gini':[], str(data[1]) + 'Gini':[], str(data[2]) + 'Gini':[]}
        TotalEdges = self.dfShape[0]
        AverageEntropyDict = {str(data[0]) + 'Entropy':[], str(data[1]) + 'Entropy':[], str(data[2]) + 'Entropy':[]}

        # print(self.X_train['Age'][38])
        # print(self.y_train[38])

        GiniValuesForEveryEdge = [[], [], []]

        for i in range(len(data)):
            PYesEdge1SmallerThen = self.X_train[str(data[i])] <= SplitValue[i]
            st.dataframe(PYesEdge1SmallerThen)
            PYesEdge1SmallerThenAndEqual = self.y_train[PYesEdge1SmallerThen] == 'absent'
            PYesEdge1SmallerThenAndEqualAndTrue = PYesEdge1SmallerThenAndEqual[PYesEdge1SmallerThenAndEqual] == True
            PYesEdge1SmallerThenAndEqualAndFalse = len(PYesEdge1SmallerThenAndEqual) - len(PYesEdge1SmallerThenAndEqualAndTrue)
            LenPYesEdge1SmallerThenAndEqualAndTrue = len(PYesEdge1SmallerThenAndEqualAndTrue)
            LenPYesEdge1SmallerThenAndEqualAndFalse = PYesEdge1SmallerThenAndEqualAndFalse
            GiniValuesForEveryEdge[i].append(LenPYesEdge1SmallerThenAndEqualAndTrue)
            GiniValuesForEveryEdge[i].append(LenPYesEdge1SmallerThenAndEqualAndFalse)

            PYesEdge1BiggerThen = self.X_train[str(data[i])] > SplitValue[i]
            PYesEdge1BiggerThenAndEqual = self.y_train[PYesEdge1BiggerThen] == 'absent'
            PYesEdge1BiggerThenAndEqualAndTrue = PYesEdge1BiggerThenAndEqual[PYesEdge1BiggerThenAndEqual] == True
            PYesEdge1BiggerThenAndEqualAndFalse = len(PYesEdge1BiggerThenAndEqual) - len(PYesEdge1BiggerThenAndEqualAndTrue)
            LenPYesEdge1BiggerThenAndEqualAndTrue = len(PYesEdge1BiggerThenAndEqualAndTrue)
            LenPYesEdge1BiggerThenAndEqualAndFalse = PYesEdge1BiggerThenAndEqualAndFalse
            GiniValuesForEveryEdge[i].append(LenPYesEdge1BiggerThenAndEqualAndTrue)
            GiniValuesForEveryEdge[i].append(LenPYesEdge1BiggerThenAndEqualAndFalse)


            if sum(GiniValuesForEveryEdge[i]) != 56 len(PYesEdge1SmallerThen):
                raise ValueError('you need to use all options in edges to calculate the Gini index!!! ')

        print(GiniValuesForEveryEdge)
        print(sum(GiniValuesForEveryEdge[0]))
        print(sum(GiniValuesForEveryEdge[1]))
        print(sum(GiniValuesForEveryEdge[2]))
            # EntropyEdge1 = - (PYesEdge1 * numpy.log2(PYesEdge1) + PNoEdge1 * numpy.log2(PNoEdge1))
            # EntropyEdge2 = - (PYesEdge2 * numpy.log2(PYesEdge2) + PNoEdge2 * numpy.log2(PNoEdge2))
            #
            # I_EntropySuB = (PYesEdge1 + PNoEdge1) / len(self.X_train <= SplitValue[i]) * EntropyEdge1 \
            #                + (PYesEdge2 + PNoEdge2) / len(self.X_train > SplitValue[i]) * EntropyEdge2
            # AverageEntropyDict[str(data[i]) + 'Entropy'].append(I_EntropySuB)

        # print(AverageEntropyDict)



        # for i in range(len(data)):
        #     Edge1, Edge2 = sum(self.X_train[str(data[i])] < SplitValue[i]) ,\
        #                    TotalEdges - sum(self.X_train[str(data[i])] < SplitValue[i])
        #     print(Edge1, Edge2)
        #     Gini = 1 - ( (Edge1 / TotalEdges) ** 2 + (Edge2 / TotalEdges) ** 2)
        #     GiniDict[str(data[i]) + 'Gini'].append(Gini)
        # print(GiniDict)
        # return GiniDict
##---------------------------Training the Decision Tree Classifier------------------------------------------------------
    def TrainingData(self):
        print(Fore.LIGHTYELLOW_EX + '\n-----------------Training the Decision Tree Classifier-----------------------\n')
        # from sklearn.tree import DecisionTreeClassifier
        dtree = DecisionTreeClassifier(criterion="gini")
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
        print(Fore.RED + '\n accuracy_score(y_test,predictions)\n', accuracy_score(self.y_test, predictions))
        # print(Fore.RED + '\n confusion_matrix(y_test, predictions)\n', confusion_matrix(self.y_test, predictions))
        # print(Fore.RED + '\nclassification_report(y_test, predictions)\n', classification_report(self.y_test, predictions))
        return predictions

TestAreTree = DecisionTrees()
