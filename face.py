
import tensorflow as tf
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt#
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.utils.np_utils import *
from keras import models
from keras.models import load_model

#讀取csv
data = pd.read_csv('train_label.csv')
X = []
for i in range(data.shape[0]):
  path = 'train/trn_img'+data['file_id'][i].astype('str').zfill(5)+'.jpg'
  img = image.load_img(path, target_size=(192, 168, 3))
  img = image.img_to_array(img)
  img = img/255.0
  X.append(img)
X = np.array(X)
y = data['label']
y = y.to_numpy()
y = to_categorical(y,41)

#切割數據
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, test_size = 0.15)
#切分訓練data和測試data

#創建model
model = Sequential()

model.add(Conv2D(16, (3,3), activation='relu', input_shape = X_train[0].shape))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.4))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(41, activation='softmax'))

#編譯驗證訓練集
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 64, epochs = 50, validation_data = (X_test, y_test))

#get loss value檢查model是否正常
#loss, accuracy = model.evaluate(X_test, y_test, batch_size = 128)
#測試print('test ')
#測試print('loss %s\n accuracy %s' % (loss, accuracy))

#將訓練好的model存起來
model.save('./CNN_Model.h5')

testdata = pd.read_csv('test.csv')
Test = []

for i in range(testdata.shape[0]):
  path = 'test/tst_img'+testdata['file_id'][i].astype('str').zfill(5)+'.jpg'
  img = image.load_img(path, target_size=(192, 168, 3))
  img = image.img_to_array(img)
  img = img/255.0
  Test.append(img)
Test = np.array(Test)

#進行預測
pred = model.predict(Test)
prediction = []
for i in range(len(pred)): 
  top = np.argmax(pred[i])#取得答案
  if top == 40:
    top = 410711225
  else:
    prediction.append(top)
  #print("Prediction {0}:{1}".format((i+1),top))

#將prediction存進csv檔
file = open('face_410711225.csv', 'w', newline='')
writer = csv.writer(file, delimiter=',')
writer.writerow(['Id', 'Category'])
for i in range(0, len(prediction)):
    writer.writerow([i+1,int(prediction[i])])
#測試print("Successful create")



