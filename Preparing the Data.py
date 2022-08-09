import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf


df = pd.read_csv('DATA/fake_reg.csv')
X = df[['feature1', 'feature2']].values
y = df['price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# X_train, X_test = MinMaxScaler().fit_transform(X_train), MinMaxScaler().fit_transform(X_test)
scalar = MinMaxScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)
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

n_epochs = 250

model2.compile(optimizer='rmsprop', loss='mse')
model2.fit(x=X_train, y=y_train, epochs=n_epochs)
##---------------------graph epochs--------------------------
y = model2.history.history['loss']
x = [i for i in range(n_epochs)]
plt.plot(x, y)
plt.savefig('Graph.png')
###------------------------predictions------------------------
test_predict = model2.predict(X_test)
test_predict = pd.Series(test_predict.reshape(300,))
pred_df = pd.DataFrame(y_test, columns=['Test True Y'])
pred_df = pd.concat([pred_df, test_predict], axis=1)
pred_df.columns = ['Test True Y', 'Model predict']
print(pred_df)
###------------------------predictions on new data wotout label------------------------
new_gem = [[998,1000]]
new_gem = scalar.transform(new_gem)                 #scale by the X_train
print(new_gem)
print(model2.predict(new_gem))
###-----------------------sav and load your model----------------------------------------

model2.save('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\')
loaded_model = tf.keras.models.load_model('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\')
loaded_model.predict(new_gem)
print(loaded_model.predict(new_gem))
# tf.saved_model.save(model2, 'C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\')
# saved_model = tf.saved_model.load('C:\\Users\\or_cohen\\PycharmProjects\\TestGithub1\\')




