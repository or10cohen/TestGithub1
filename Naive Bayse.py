import pandas as pd
import numpy as np
import random
import colorama
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style
colorama.init(autoreset = True)


DataBase = pd.read_csv('C:\\Users\\or_cohen\\Desktop\\pyt\\diabetes.csv')
print(Fore.RED + '\nThe shape of DataBase are: ', DataBase.shape)

print(Fore.RED + '\nsample of 3 first rows data\n', DataBase.head(3))
print(Fore.RED + '\nsample from 20-22 rows data\n',DataBase[20:23])
print(Fore.RED + '\nsample from column Age and 20-22 rows data\n', DataBase.Age[20:23])

