import pandas as pd
import numpy as np
from numpy.random import randn
import random
import colorama
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style
colorama.init(autoreset = True)

##------------------------------------------lecture 1 Series-----------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------
# labels = ['a', 'b', 'c']
# MyData = [10, 20, 30]
# array = np.array(MyData)
# d = {'a':10, 'b':20, 'c':30}
#
# print('\nlabels: ', labels,'\t\t\t\t\t\t\ttype:', type(labels),'\nMydata: ', MyData,
# '\t\t\t\t\t\t\t\ttype:', type(MyData),'\nMyData array: ', array,'\t\t\t\t\t\t\ttype:', type(array),'\ndictionary: ',
# d, '\t\t\ttype:', type(d),'\n\n')


# series1 = pd.Series(data=MyData)
# print('series1 use command: pd.Series(data=MyData):\n', series1)
#
# series2 = pd.Series(data=MyData, index=labels)
# print('\n\nseries2 use command: pd.Series(data=MyData, index=labels):\n', series2)
# # series2 = pd.Series(MyData, labels)               # same like before
# # print('\n\nseries1 use command: pd.Series(data=MyData, index=labels):\n', series2)
#
# series3 = pd.Series(data=array)
# print('\n\nseries3 same like series1 but use array now instead list, use command: pd.Series(data=array):\n'
#       'so as we see, we can use array in pd.Series.\n', series3)
#
# series4 = pd.Series(d)
# print('\n\nseries3 same like series2 but use dictionary now instead 2 list, use command: pd.Series(d):\n'
#       'so as we see, we can use dictionary in pd.Series.\n', series4)

##----------------------------------------------------------------------------------------------------------------------

# ser1 = pd.Series([1,2,3,4],['USA','Germany','Russian','Israel'])    #numbers are the Data and countrios names are the labels
# print('\nwe create a new Series use: pd.Series([1,2,3,4],[\'USA\',\'Germany\',\'Russian\',\'Israel\'])\n', ser1)
#
# print('\n\nwe have two different ways to show data per index  label: \n1. ser1.USA:\t\t\t', ser1.USA,
#       '\n2. ser1[\'USA\']:\t\t\t', ser1['USA'])

##------------------------------------------lecture 2 DataFrames - Part 1-----------------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------------------
#
# np.random.seed(101)             # to use same random as the video
# df = pd.DataFrame(randn(5,4), ['A','B','C','D','E'],['W','X','Y','Z'])             # to see hint ues ctrl + p
# print(Fore.RED + '\nwe build a new DataFrame use the command pd.DataFrame(Data argument, index, columns).')
# print('for the Data argument we use randn(5, 4).'
#       '\nfor the index(rows) we use list: [\'A\',\'B\',\'C\',\'D\',\'E\']\nfor the columns we use" [\'W\',\'X\',\'Y\',\'Z\']\n\n')
# print(Fore.RED + 'and are Data frame looks:\n', df)
#
# print(Fore.RED + '\nto see shape of df use df.shape:  ', df.shape)
# print(Fore.RED + 'so as we see we have {} rows ans {} columns' .format(df.shape[0], df.shape[1]))
#
# print(Fore.RED + '\n\nwe can print any column we want, like column W use: df.W or df[\'W\']:\n', df.W)
# print(Fore.RED + '\n\nwe can print multiple columns, like column W and X use LIST like : df[[\'W\', \'X\']]:\n', df[['W','X']])
#
# print(Fore.RED + '\n\nwe can print any rows we want, like row A use: df.loc[\'A\']:\n', df.loc['A'])
# print(Fore.RED + '\n\nwe can print any rows use the row index we want, like row A use: df.iloc[0] *pay attention about the diffrent!*:\n', df.iloc[0])
#
# print(Fore.RED + '\n\nwe can print multiple columns, like column W and X use LIST like : df[[\'W\', \'X\']]:\n', df[['W','X']])
#
# print(Fore.RED + '\n\nwe can print specific cell 1 row & 1 culomn , like row B and column Y: df.ioc[\'B\',\'Y\']:\n', df.loc['B','Y'])
# print(Fore.RED + '\n\nfor print specific multiple rows&aculomns, like row A-B and column X-Y use list like: df.ioc[[,\'A\' \'B\'],[\'X\', \'Y\']]"\n', df.loc[['A','B'], ['X','Y']])
#
# df['NEW'] = df['W'] + df['X']
# print(Fore.RED + '\ncreate new column: df[\'NEW\'] = df[\'W\'] + df[\'X\']:\n', df['NEW'])
# print(Fore.RED + '\nnow we have new column in Dataframe:\n', df)
#
# df.drop('X', axis=1, inplace=True)      #axis=0 mean the rows and axis=1 for the columns    #interplace=True for drop for the original df and not save the column drop Data.
# print(Fore.RED + '\ndrop any column you want use df.drop(\'X\', axis=1, inplace=True). '
#                  '\naxis=0(default) mean the rows and axis=1 for the columns.'
#                  '\ninterplace=True for drop from the original df and not save the column we drop.\n', df)
#
# df.drop('A', axis=0, inplace=True)      #axis=0 mean the rows and axis=1 for the columns    #interplace=True for drop for the original df and not save the column drop Data.
# print(Fore.RED + '\nwe can drop rows too use df.drop(\'A\', axis=0, inplace=True). '
#                  '\naxis=0(default) mean the rows and axis=1 for the columns.'
#                  '\ninterplace=True for drop from the original df and not save the row we drop Data.\n', df)
##------------------------------------------lecture 3 DataFrames - Part 2-----------------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------------------
# np.random.seed(101)             # to use same random as the video
# df = pd.DataFrame(randn(5,4), ['A','B','C','D','E'],['W','X','Y','Z'])             # to see hint ues ctrl + p
# print(Fore.RED + '\nwe build a new DataFrame use the command pd.DataFrame(Data argument, index, columns).')
# print('for the Data argument we use randn(5, 4).'
#       '\nfor the index(rows) we use list: [\'A\',\'B\',\'C\',\'D\',\'E\']\nfor the columns we use" [\'W\',\'X\',\'Y\',\'Z\']\n\n')
# print(Fore.RED + 'and are Data frame looks:\n', df)
#
# print(Fore.RED + '\ncheck True/False or any conditional selection you want when use the command: df > 0\n', df > 0)
# print(Fore.RED + '\nan other way to show just the positive value: df[df > 0]\n', df[df > 0])
# print(Fore.RED + '\nuse True/False in chosen column: df[\'W\'] > 0\n', df['W'] > 0)
# print(Fore.RED + '\nlike before, we want to see just the positive value, just now because we chose the W column,\n'
#                  'we dont gonna see NaN and the all(!) row when negative value in rows W was deleted: df[df[\'W\'] > 0] \n', df[df['W']>0])
#
# resultdf = df[df['W'] > 0]
# print(Fore.RED + '\nto see just on column, like X, but delete the row when in W(!) the value bigger then 0, we save he command first:\nresultdf =  df[df[\'W\']>0]'
#       'and after to see just the the column X, we use the command: resultdf[\'X\']\n', resultdf['X'])
# print(Fore.RED + 'in an other way but some: df[df[\'W\'] > 0][\'X\']\n', df[df['W'] > 0]['X'])
# print(Fore.RED + 'or from multiply columns(see the double[[]]): df[df[\'W\'] > 0][[\'X\', \'Y\']]\n', df[df['W'] > 0][['X', 'Y']])
# print(Fore.RED + 'we can use multiply conditions with the command(pay attenation use & and not \'and\', for \'or\' use \'|\' ): df[(df[\'W\'] > 0) & (df[\'W\'] < 1))]\n', df[(df['W'] > 0) & (df['W'] < 1)])
#
# print(Fore.RED + '\nif we want use the index, and move it to the first row in data(now the index is a numbers extand and the title row of index is \'index\'). use the command: df.reset_index():\n'
#                  'note: same like before the change dont save in the original df, from saving use:df.reset_index(inplace=True)\n', df.reset_index())
# newindex = ['Haifa', 'Jeruslem', 'TLV', 'Aco', 'Kryot']    # you can use: newindex = 'Haifa Jeruslem TLV Aco Kryot'.split
# df['Cities'] = newindex
#
# print(Fore.RED + '\nif we want set other index from are row in the data, we can use the command: df.set_index(\'are row we want to use as index\')\n'
#                  'first we add column in the commands: newindex = [\'Haifa\', \'Jeruslem\', \'TLV\', \'Aco\', \'Kryot\']   and    df[Cities] = newindex\n'
#                  'and after we use the command: df.set_index(\'Cities\')\n'
#                  'note: same like before the change dont save in the original df, from saving use:df.reset_index(inplace=True)\n', df.set_index('Cities'))

##------------------------------------------lecture 4 DataFrames - Part 3-----------------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------------------


