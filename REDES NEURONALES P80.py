# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:07:00 2019

@author: Roberth Prz
"""
#MODELO RNA P80
#--------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
#from keras_lr_finder.lr_finder import LRFinder
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot
import matplotlib.pyplot as plt
# Importing the dataset
data_file_path='C:/Users/Roberth Prz/Documents/Python Scripts/data1.csv'

home_data=pd.read_csv('C:/Users/Roberth Prz/Documents/Python Scripts/data1.csv')

#Correlation between variables
home_data.corr(method="pearson")
home_data1=home_data.drop(columns=['Diameter','Bench','Overdrilling'])
home_data1.corr(method="pearson")

#create target objet and call it y


X=home_data1.drop(columns=['P80','P50','P20'])

y = home_data1.drop(columns=['B','S','GU','Density','Stemming','D. Expl','Kg. Expl','Powder Factor','P50','P20'])
#y=home_data1['P80']
print(y)
print(X)
# standardize dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.fit(home_data))
print(scaler.transform(home_data))

X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

# split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

print(X_test)


# define model
model = Sequential()
model.add(Dense(13, input_dim=8, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))
opt = SGD(lr=0.05, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer=opt)
# fit model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500, verbose=0)
# evaluate the model
train_mse = model.evaluate(X_train, y_train, verbose=0)
test_mse = model.evaluate(X_test, y_test, verbose=0)
print (train_mse," ",test_mse)



# plot loss during training
pyplot.title('Loss / Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
pyplot.legend()
pyplot.show()
#-------------------------------------
#Guardar la red y pesos sinapticos

#import h5py
#h5py.run_tests()
#pip install h5py
#from keras.models import model_from_json
#import os
#guardar el modelo de red a JSON
#model_json=model.to_json()
#with open("model.json","w")as json_file:
 #       json_file.write(model_json)
#guardar pesos sinapticos  a HDF5
#model.save_weights("model.h5")
#print("Saved model to disk")

#Cargar json y crear modelo
#json_file=open('model.json','r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
#cargar pesos sinapticos en el nuevo modelo
#loaded_model.load_weights("model.h5")
#print("loaded model from disk")
#Evaluando modelo cargado en testing data
#loaded_model.compile(loss='mean_squared_error', optimizer=opt)
#score = loaded_model.evaluate(X_real_train, y_real_train)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score))
#print( u'Estadístico R_2: %.2f' % r2_score(y, X))
#print(loaded_model.score(X_real_train, y_real_train))
#print(score)

#save model and architecture to single file
model.save("model-rnap80.h5")
print("saved model to disk")
#load the model
from keras.models import load_model
model_1=load_model('model-rnap80.h5')
#summarize model
model_1.summary()

#evaluate the model
train_mse1 = model_1.evaluate(X_train, y_train, verbose=0)
test_mse1 = model_1.evaluate(X_test, y_test, verbose=0)
print (train_mse1," ",test_mse1)

#--------------------------------------
#ANALISIS MODELO RNA CON TESTING DATA

#invertir la data escalada
X_real=scaler.inverse_transform(X_test)
y_real=scaler.inverse_transform(y_test)

X_real_train=scaler.inverse_transform(X_train)

print(X_real)
print(y_real)
# plot ypred and ytest
y_pred_test_sintr = model_1.predict(X_test)
y_pred_test=scaler.inverse_transform(y_pred_test_sintr)

plt.plot(y_real, color = 'red', label = 'Real data')
plt.plot(y_pred_test, color = 'blue', label = 'Predicted data')
plt.title('Gráfica Comparativa P80')
plt.legend()
plt.show()


print(y_real)
print(y_pred_test)
print(X_test)

# calculamos el coeficiente de determinación R2
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

modelo=linear_model.LinearRegression()

modelo.fit(y_real,y_pred_test)

# Ahora puedo obtener el coeficiente b_1
print (u'Coeficiente beta1: ', modelo.coef_[0])

# Podemos predecir usando el modelo
y_predic = modelo.predict(y_real)
 

# Por último, calculamos el error cuadrático medio y el estadístico R^2
print( u'Error cuadrático medio: %.2f' % mean_squared_error(y_real, y_pred_test))
print( u'Estadístico R_2: %.2f' % r2_score(y_real, y_pred_test))

#analisis estadistico 
media_y_real = y_real.mean()
media_y_pred_test= y_pred_test.mean()
std_y_real = y_real.std (ddof=0)
std_rna_test = y_pred_test.std (ddof=0)
cv_y_real=std_y_real/media_y_real
cv_rna_test=std_rna_test/media_y_pred_test


print("X80 media Testing data : %f" % (media_y_real))
print("X80 media Modelo RNA : %f" % (media_y_pred_test))
print("Std.Desv Testing Data : %f" % (std_y_real))
print("Std.Desv modelo RNA : %f" % (std_rna_test))
print("C.V.Testing Data : %f" % (cv_y_real))
print("C.V.RNA : %f" % (cv_rna_test))

#---------------------------------------------------------------
#ANALISIS MODELO RNA CON TRAINING DATA

#invertir la data escalada
X_real=scaler.inverse_transform(X_test)
y_real=scaler.inverse_transform(y_test)

X_real_train=scaler.inverse_transform(X_train)
y_real_train=scaler.inverse_transform(y_train)
print(X_real_train)
print(y_real_train)
# plot ypred and ytest
y_pred = model_1.predict(X_train)
y_pred_train=scaler.inverse_transform(y_pred)

plt.plot(y_real_train, color = 'red', label = 'Real data')
plt.plot(y_pred_train, color = 'blue', label = 'Predicted data')
plt.title('Gráfica Comparativa P80')
plt.legend()
plt.show()


print(y_real)
print(y_pred_train)
print(X_test)

# calculamos el coeficiente de determinación R2
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

modelo=linear_model.LinearRegression()

modelo.fit(y_real_train,y_pred_train)

# Ahora puedo obtener el coeficiente b_1
print (u'Coeficiente beta1: ', modelo.coef_[0])

# Podemos predecir usando el modelo
#y_predic = modelo.predict(y_real_train)
 

# Por último, calculamos el error cuadrático medio y el estadístico R^2
print( u'Error cuadrático medio: %.2f' % mean_squared_error(y_real_train, y_pred_train))
print( u'Estadístico R_2: %.2f' % r2_score(y_real_train, y_pred_train))

#analisis estadistico 
media_y_real_train = y_real_train.mean()
media_y_pred_train= y_pred_train.mean()
std_y_real_train = y_real_train.std (ddof=0)
std_rna_train = y_pred_train.std (ddof=0)
cv_y_real_train=std_y_real_train/media_y_real_train
cv_rna_train=std_rna_train/media_y_pred_train


#cv = df["nota"].std(ddof=0) / df["nota"].mean()

print("X80 media real : %f" % (media_y_real_train))
print("X80 media Modelo RNA : %f" % (media_y_pred_train))
print("Std.Desv Training Data : %f" % (std_y_real_train))
print("Std.Desv modelo RNA : %f" % (std_rna_train))
print("C.V.Training Data : %f" % (cv_y_real_train))
print("C.V.RNA : %f" % (cv_rna_train))

#---------------------------------------------------------
#MODELO DE REGRESION MULTIVARIABLE

#defino algoritmo modelo de regresion lineal multiple a usar
lr_multiple=linear_model.LinearRegression()

#entreno el modelo con los datos X_real_train e y_real_train
lr_multiple.fit(X_real_train, y_real_train)

#realizo una predicción con los datos de testing
y_pred_test_multiple=lr_multiple.predict(X_real)
#realizo una predicción con los datos de training
y_pred_train_multiple=lr_multiple.predict(X_real_train)
#valores de pendiente e interseccion
print('DATOS DEL MODELO DE REGRESION MULTIPLE')
print()
print('Valor de las pendientes o coeficientes "a"')
print(lr_multiple.coef_)
print('Valor de la intersección o coeficiente "b"')
print(lr_multiple.intercept_)

#Precisión del modelo de regresión múltiple con training data 
print(lr_multiple.score(X_real_train, y_real_train))
#Precisión del modelo de regresión múltiple con training data
print( u'Estadístico R_2: %.2f' % r2_score(y_real_train, y_pred_train_multiple))

#Precisión del modelo de regresión múltiple con testing data
print( u'Estadístico R_2: %.4f' % r2_score(y_real, y_pred_test_multiple))

#Analisis estadistico del modelo de regresion multiple 

#con testing data

media_y_real = y_real.mean()
media_y_pred_test_multiple= y_pred_test_multiple.mean()
std_real_test = y_real.std (ddof=0)
std_pred_test_multiple = y_pred_test_multiple.std (ddof=0)
cv_real_test=std_real_test/media_y_real
cv_test_multiple=std_pred_test_multiple/media_y_pred_test_multiple

print("X80 media testing data : %f" % (media_y_real))
print("X80 pred multiple : %f" % (media_y_pred_test_multiple))
print("Std.Desv testing data : %f" % (std_real_test))
print("Std.Desv modelo regresión múltiple : %f" % (std_pred_test_multiple))
print("C.V.real de testing data : %f" % (cv_real_test))
print("C.V.modelo regresión múltiple : %f" % (cv_test_multiple))


#con training data

media_y_real_train = y_real_train.mean()
media_y_pred_train_multiple= y_pred_train_multiple.mean()
std_real_train = y_real_train.std (ddof=0)
std_pred_train_multiple= y_pred_train_multiple.std (ddof=0)
cv_real_train=std_real_train/media_y_real_train
cv_train_multiple=std_pred_train_multiple/media_y_pred_train_multiple


print("X80 media training data : %f" % (media_y_real_train))
print("X80 pred modelo regresión multiple : %f" % (media_y_pred_train_multiple))
print("Std.Desv training data : %f" % (std_real_train))
print("Std.Desv modelo regresión múltiple : %f" % (std_pred_train_multiple))
print("C.V.real de training data : %f" % (cv_real_train))
print("C.V.modelo regresión múltiple : %f" % (cv_train_multiple))
#-------------------------------------------------------------------------
#Gráfica de comparación lineal 
y_pred_test_multiple1=sorted(y_pred_test_multiple)
print(y_pred_test_multiple1)
y_real1=sorted(y_real)
print(y_real1)
y_pred_test1=sorted(y_pred_test)
print(y_pred_test1)



plt.plot(y_pred_test_multiple1, y_real1, color = 'red')
plt.plot(y_pred_test_multiple1, y_real1, color = 'blue')
plt.title('Modelo de Regresion lineal Multiple vs P80 (Testing Data)')
plt.xlabel('P80 predicted MRLM')
plt.ylabel('P80 Testing Data ')
plt.show()

plt.plot(y_pred_test_multiple1 , 'b.', y_pred_test1, 'rd', y_real1, 'g^')

#gráfica comprativa modelos de Testing

y_pred_test_multiple1=sorted(y_pred_test_multiple)
print(y_pred_test_multiple1)
y_real1=sorted(y_real)
print(y_real1)
y_pred_test1=sorted(y_pred_test)
print(y_pred_test1)

plt.plot(y_real1, marker="x", linestyle=':', color='b', label="P80")
plt.plot(y_pred_test1, marker='*', linestyle='-', color='g', label ="ANN Model $R^{2}$=0.81")
plt.plot(y_pred_test_multiple1, marker='o', linestyle='--', color='r', label="MLR Model $R^{2}$=0.78")
plt.legend(loc="upper left")
plt.title('P80 Breakage vs ANN Model vs Multiple Linear Regression Model (Testing Data)')

#gráfica comprativa modelos de training

y_pred_train_multiple1=sorted(y_pred_train_multiple)
print(y_pred_train_multiple1)
y_real_train1=sorted(y_real_train)
print(y_real_train1)
y_pred_train1=sorted(y_pred_train)
print(y_pred_train1)

plt.plot(y_real_train1, marker="x", linestyle=':', color='b', label="P80")
plt.plot(y_pred_train1, marker='*', linestyle='-', color='g', label ="ANN Model $R^{2}$=0.87")
plt.plot(y_pred_train_multiple1, marker='o', linestyle='--', color='r', label="MLR Model $R^{2}$=0.85")
plt.legend(loc="upper left")
plt.title('P80 Breakage vs ANN Model vs Multiple Linear Regression Model (Training Data)')








