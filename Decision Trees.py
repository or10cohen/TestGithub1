import numpy as np
import pandas as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset = True)


df = pd.read_csv('kyphosis.csv')            #import data
print(Fore.RED + '\nfirst we can print the first 5 rows in data use command: df.head()\n', df.head())
print(Fore.RED + '\nwe check the size, columns title, no. of row and more info data use command: df.info()')
print(df.info())

## sns.pairplot(df, hue='Kyphosis')

print(Fore.RED + '\nSplit the data to Train/Test: in the first time we use sklearn library.\n'
                 'To split the data we use the library: from sklearn.model_selection import train_test_split'
                 , Fore.BLUE + '\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)')


print(Fore.RED + '\nFor X(data) we use the all data *without* the label, column \'Kyphosis\'. use the command:'
      , Fore.BLUE + 'X = df.drop(\'Kyphosis\', axis=1)  NOTE: axis=1 for drop column\n', df.drop('Kyphosis', axis=1)
      , Fore.RED + '\nFor y(label) we use just the \'Kyphosis\' column. use the command:'
      , Fore.BLUE + 'y = df[\'Kyphosis\']\n', df['Kyphosis'] )

X = df.drop('Kyphosis', axis=1)
y = df['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# dtree = DecisionTreeClassifier()
# dtree.fit()