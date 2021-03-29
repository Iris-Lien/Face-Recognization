import tkinter as tk
import tensorflow as tf
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.utils.np_utils import *
from keras import models
from keras.models import load_model
from tkinter import ttk
from tkinter import simpledialog as sd
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

#取得檔案位置
def path():
    global Path
    Path = filedialog.askopenfilename()
    print("Path:{}".format(Path))
    img = Image.open(Path)
    img = img.resize((192, 168), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(window, image=img)
    panel.image = img
    panel.pack(padx=10,pady=100)
    
#載入model和圖片進行辨識
def reco():
    model = load_model('./CNN_Model.h5')
    img = image.load_img(Path, target_size=(192, 168, 3))
    img = image.img_to_array(img)
    img = img/255.0
    Test = []
    Test.append(img)
    Test = np.array(Test)
    pred = model.predict(Test)
    for i in range(len(pred)):  
        top = np.argmax(pred[i])
        if(top == 40): 
            print("Prediction:{0}".format("410711225"))
            t = "Prediction:"+str("410711225")
        else:
            print("Prediction:{0}".format(top))
            t = "Prediction:"+str(top)
    label= tk.Label(window,text = t,bg = '#95CACA').pack()
    
#建立視窗、設定視窗名稱大小等
window=tk.Tk()
window.title('Face Recognize')
window.geometry('500x500')
window.resizable(0,0)
window.configure(background='white')

#加入按鈕(選擇要辨識的圖片、進行預測)
ttk.Button(window, text="Select Image", command=path).pack(side=tk.BOTTOM)
ttk.Button(window, text="Predict", command=reco).pack(side=tk.BOTTOM)

#不斷執行視窗程式
window.mainloop()
