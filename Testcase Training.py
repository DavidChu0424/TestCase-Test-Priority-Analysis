# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

print("Code by David_Chu")
print("請選擇 Training 檔案")
root = tk.Tk()
root.withdraw()
fileread= filedialog.askopenfilename()

#讀取資料
dataset = pd.read_csv(fileread , encoding = "big5")

#填補缺失資料
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missingv_values = "Nan" , stragtey = "mean" , axis = 0)


X = dataset.iloc[:,:-1].values #排除最一行目標資料
y = dataset.iloc[:,7].values #選擇最後一行目標資料

X #顯示 X 資料

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])#將第0列轉換成數字型態
X[:,1] = labelencoder_X.fit_transform(X[:,1])#將第1列轉換成數字型態
X[:,4] = labelencoder_X.fit_transform(X[:,4])#將第1列轉換成數字型態

y#顯示 y 資料

"""onehotencoder = OneHotEncoder(categories = 'auto') #categorical_features = [0] 表示哪一列是分類的資料
X = onehotencoder.fit_transform(X).toarray()


labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)"""

#分割測試集及訓練集
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 2, random_state = 0)

#特徵縮放_歸一化(樹形演算法不須特徵縮放)
#from sklearn.preprocessing import MinMaxScaler
#min_max_scaler = MinMaxScaler()
#X_train_minmax = min_max_scaler.fit_transform(X_train)
#X_test_minmax = min_max_scaler.transform(X_test)


#決策樹涵式
trees = DecisionTreeClassifier(criterion = 'entropy' , max_depth = 20, random_state = 0)
trees.fit(X_train, y_train) #Decision tree 訓練
trees.predict(X_test)        #Decision tree 預測 X_test測試集
"""print(y_test)                      #顯示 y_test 測試集"""
"""print(trees.score(X_test, y_test))  #測試集預測評分"""
"""tree.plot_tree(trees)"""

#建模
trees.fit(X,y)
joblib.dump(trees,'TestCaseAnalytics.pkl')

