# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tkinter as tk
from tkinter import filedialog
import pandas as pd
from sklearn.externals import joblib
from sklearn import tree


#讀取資料
print("Code by David_Chu")
root = tk.Tk()
root.withdraw()

print("請選擇 Model 檔案")
Modelread = filedialog.askopenfilename()
print("請選擇 Testing 檔案")
fileread = filedialog.askopenfilename()

dataset = pd.read_csv(fileread , encoding = "big5")

#填補缺失資料
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missingv_values = "Nan" , stragtey = "mean" , axis = 0)


X = dataset.iloc[:,:-1].values #排除最一行目標資料

X #顯示 X 資料

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])#將第0列轉換成數字型態
X[:,1] = labelencoder_X.fit_transform(X[:,1])#將第1列轉換成數字型態
X[:,4] = labelencoder_X.fit_transform(X[:,4])#將第1列轉換成數字型態


treemodel = joblib.load(Modelread)
result = treemodel.predict(X[0:])
print("檔案生成，請查看 Test case predict result")
pdresult = pd.DataFrame(result)
dataset.insert(8,'Predictresult',pdresult)
dataset.to_excel('Test case predict result.xls') 

