import pandas as pd
import numpy as np
from numpy.random import randn
import random
import colorama
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style
colorama.init(autoreset = True)

#------------------------------------------lecture 1 Series-----------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
# labels = ['a', 'b', 'c']
# MyData = [10, 20, 30]
# array = np.array(MyData)
# d = {'a':10, 'b':20, 'c':30}
#
# print('\nlabels: ', labels,'                        type:', type(labels),'\nMydata: ', MyData,
# '                           type:', type(MyData),'\nMyData array: ', array,'                       type:', type(array),'\ndictionary: ', d, '        type:', type(d),'\n\n')


#----------------------------------------------------------------------------------------------------------------------

# series = pd.Series(data=MyData)
# print('\n\nseries use command: pd.Series(data=MyData):\n', series)
#
# series1 = pd.Series(data=MyData, index=labels)
# print('\n\nseries1 use command: pd.Series(data=MyData, index=labels):\n', series1)
# # series1 = pd.Series(MyData, labels)               # same like before
# # print('\n\nseries1 use command: pd.Series(data=MyData, index=labels):\n', series1)
#
# series2 = pd.Series(data=array)
# print('\n\nseries2 same like series but use array now instead list, use command: pd.Series(data=array):\n', series2)
#
# series3 = pd.Series(d)
# print('\n\nseries3 same like series1 but use dictionary now instead 2 list, use command: pd.Series(d):\n', series3)

#----------------------------------------------------------------------------------------------------------------------

# ser1 = pd.Series([1,2,3,4],['USA','Germany','Russian','Israel'])    #numbers are the Data and countrios names are the labels
# print('we create a new Series use: pd.Series([1,2,3,4],[''USA'',''Germany'',''Russian'',''Israel''])\n', ser1)
#
# print('\n\nwe have two different ways to show data per index  label: \n1. ser1.USA:                ', ser1.USA,'\n2. ser1[''USA'']:               ', ser1['USA'])

#------------------------------------------lecture 2 DataFrames - Part 1-----------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------

np.random.seed(101)             # to use same random as the video
df = pd.DataFrame(randn(5,4), ['A','B','C','D','E'],['W','X','Y','Z'])             # to see hint ues ctrl + p
print(Fore.RED + '\nwe build a new DataFrame use the command pd.DataFrame(Data argument, index, columns).')
print('for the Data argument we use randn(5, 4).'
      '\nfor the index(rows) we use list: [\'A\',\'B\',\'C\',\'D\',\'E\']\nfor the columns we use" [\'W\',\'X\',\'Y\',\'Z\']\n\n')
print(Fore.RED + 'and are Data frame looks:\n', df)

print(Fore.RED + '\nto see shape of df use df.shape:  ', df.shape)
print(Fore.RED + 'so as we see we have {} rows ans {} columns' .format(df.shape[0], df.shape[1]))

print(Fore.RED + '\n\nwe can print any column we want, like column W use: df.W or df[\'W\']:\n', df.W)
print(Fore.RED + '\n\nwe can print multiple columns, like column W and X use LIST like : df[[\'W\', \'X\']]:\n', df[['W','X']])

print(Fore.RED + '\n\nwe can print any rows we want, like row A use: df.loc[\'A\']:\n', df.loc['A'])
print(Fore.RED + '\n\nwe can print any rows use the row index we want, like row A use: df.iloc[0] *pay attention about the diffrent!*:\n', df.iloc[0])

print(Fore.RED + '\n\nwe can print multiple columns, like column W and X use LIST like : df[[\'W\', \'X\']]:\n', df[['W','X']])

print(Fore.RED + '\n\nwe can print specific cell 1 row & 1 culomn , like row B and column Y: df.ioc[\'B\',\'Y\']:\n', df.loc['B','Y'])
print(Fore.RED + '\n\nfor print specific multiple rows&aculomns, like row A-B and column X-Y use list like: df.ioc[[,\'A\' \'B\'],[\'X\', \'Y\']]"\n', df.loc[['A','B'], ['X','Y']])






df['NEW'] = df['W'] + df['X']
print(Fore.RED + '\ncreate new column: df[\'NEW\'] = df[\'W\'] + df[\'X\']:\n', df['NEW'])
print(Fore.RED + '\nnow we have new column in Dataframe:\n', df)

df.drop('X', axis=1, inplace=True)      #axis=0 mean the rows and axis=1 for the columns    #interplace=True for drop for the original df and not save the column drop Data.
print(Fore.RED + '\ndrop any column you want use df.drop(\'X\', axis=1, inplace=True). '
                 '\naxis=0(default) mean the rows and axis=1 for the columns.'
                 '\ninterplace=True for drop from the original df and not save the column we drop.\n', df)

df.drop('A', axis=0, inplace=True)      #axis=0 mean the rows and axis=1 for the columns    #interplace=True for drop for the original df and not save the column drop Data.
print(Fore.RED + '\nwe can drop rows too use df.drop(\'A\', axis=0, inplace=True). '
                 '\naxis=0(default) mean the rows and axis=1 for the columns.'
                 '\ninterplace=True for drop from the original df and not save the row we drop Data.\n', df)

