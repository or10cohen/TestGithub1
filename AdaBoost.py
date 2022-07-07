import numpy as np
import pandas as pd
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset = True)

heart_disease = pd.read_csv('heart.csv')

print(Fore.RED + '\nheart_disease.head()\n', heart_disease.head())
print(Fore.RED + '\ncolumns\n', heart_disease.columns)
print(Fore.RED + '\ndescribe\n', heart_disease.describe())

class AdaBoost():

    def __init__(self):
        pass

