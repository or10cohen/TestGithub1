import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf



df = pd.read_csv('DATA/fake_reg.csv')
X = df[['feature1', 'feature2']].values
y = df['price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_test = MinMaxScaler().fit_transform(X_train), MinMaxScaler().fit_transform(X_test)
####------------creat NN option1--------------------------
model1 = tf.keras.Sequential([tf.keras.Dense(4, activation='relu'),
                    tf.keras.Dense(3, activation='relu'),
                    tf.keras.Dense(3, activation='relu'),
                    tf.keras.Dense(1)])                         #output layer
####------------creat NN option--------------------------
model2 = tf.keras.Sequential()
model2.add(tf.keras.Dense(4, activation='relu'))
model2.add(tf.keras.Dense(4, activation='relu'))
model2.add(tf.keras.Dense(4, activation='relu'))
model2.add(tf.keras.Dense(1))                                  #output layer

model2.compile(optimizer='rmsprop', loss='mse')


