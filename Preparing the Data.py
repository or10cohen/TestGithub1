import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


df = pd.read_csv('DATA/fake_reg.csv')
X = df[['feature1', 'feature2']].values
y = df['price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_test = MinMaxScaler().fit_transform(X_train), MinMaxScaler().fit_transform(X_test)
####------------creat NN option1--------------------------
# model1 = tf.keras.Sequential([tf.keras.Dense(4, activation='relu'),
#                     tf.keras.layers.Dense(3, activation='relu'),
#                     tf.keras.layers.Dense(3, activation='relu'),
#                     tf.keras.layers.Dense(1)])                         #output layer
####------------creat NN option-----------------------------
model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.Dense(4, activation='relu'))
model2.add(tf.keras.layers.Dense(4, activation='relu'))
model2.add(tf.keras.layers.Dense(4, activation='relu'))
model2.add(tf.keras.layers.Dense(1))                                  #output layer

model2.compile(optimizer='rmsprop', loss='mse')
model2.fit(x=X_train, y=y_train, epochs=250)
##---------------------graph epochs--------------------------
y = model2.history.history['loss']
x = [i for i in range(250)]
plt.plot(x, y)
plt.savefig('Graph.png')
###------------------------predictions------------------------
test_predict = model2.predict(X_test)
test_predict = pd.Series(test_predict.reshape(300,))
pred_df = pd.DataFrame(y_test, columns=['Test True Y'])
pred_df = pd.concat([pred_df, test_predict], axis=1)
pred_df.columns = ['Test True Y', 'Model predict']
print(pred_df)





