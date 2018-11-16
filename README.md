# NTU_HYLee_MachineLearning_Homework

## 事前準備

### 使用虛擬環境

```
$ virtualenv ./ENV
$ source ./ENV/bin/active
```

假設想要離開虛擬環境，則

```
$ deactivate
```

### 安裝相依套件

在虛擬環境下，安裝相依套件

```
$ pip2.7 install -r requirements.txt
```

## HW00

Original: [https://docs.google.com/presentation/d/1VCnqWX469V4Qi_dHJIXxWMhvtg3h6pyBuRYPao-GZ1k/edit#slide=id.g17e2c04840_1_14](https://docs.google.com/presentation/d/1VCnqWX469V4Qi_dHJIXxWMhvtg3h6pyBuRYPao-GZ1k/edit#slide=id.g17e2c04840_1_14)

**Q1. 輸入hw0_data.dat，請將指定column由小排到大並印出來到ans1.txt**

**A1:** Q1.sh, q1.py

**Q2. 輸入一張圖，將圖上下顛倒，左右相反（旋轉180度），並輸出到ans2.png**

**A2:** Q2.sh, q2.py

## HW01

Original: [http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/hw1.pdf](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/hw1.pdf)

本次作業使用豐原站的觀測記錄，分成train set跟test set，train set是豐原站每個月的前20天所有資料。test set則是從豐原站剩下的資料中取樣出來。

train.csv: 每個月前20天的完整資料。

test_X.csv: 從剩下的10天資料中取樣出連續的10小時為一筆，前九小時的所有觀測數據當作feature，第十小時的PM2.5當作answer。一共取出240筆不重複的test data，請根據feauure預測這240筆的PM2.5。

**Ans:**

GradientDescent class

```python
class GradientDescent(object):
    def __init__(self):
        self.__wt = None
        self.__b = None
        self.__X = None
        self.__y = None
    @property
    def wt(self):
        #...
    @property
    def b(self):
        #...
    def train_by_pseudo_inverse(self,X,y,alpha=0,validate_data=None):
        #...
    def train(self,X,y,init_wt=np.array([]),init_b=0,rate=0.01,alpha=0,epoch=1000,batch=None,validate_data=None):
        #...
    def update(self,X,y,wt,b,rate,alpha):
        #...
    def predict(self,X):
        #...
    def err_insample(self):
        #... 
    def err(self,X,y):
        #...
    def _check_data(self,X,y):
        #...
```

Using GradientDescent class to do linear regression

```python
### preproc data
# remove column "Site"
# melt "Hour" to column
# generate "Datetime"
# replace NR to 0
# change "Value" type
# pivot 'Item' to columns

### obtain training set and validation set

def gen_regression_form(df):
    #...

# record the order of columns

### gradient descent

### testing
# record ordfer of test.ID
# replace NR to 0
# ['ID','Item','Hour','Value'] form
# combine 'Hour' and 'Item' to 'Col'
# pivot 'Col' to columns
# re-order
# predict
# output
```

## HW02

Orignal: [http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/hw2.pdf](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/hw2.pdf)

**Q1: 實作Logistic Regression偵測垃圾信件**

`spam_data/spam_train.csv`裡有一些被Label過後的信件，前面幾個Columns代表一些字詞相關的Features，最後一個Column標示這一封信是否為垃圾信件。請用Logistic Regression訓練，並預測`spam_data/spam_test.csv`的每一筆信件（給予Features）是否為垃圾信？

**Ans1:**

Training at `logistic_regression_train.py`

```python
class LogisticRegression(object):
    def __init__(self,input_dim,output_dim):
        self.__dim = (input_dim,output_dim)
        self.__W = np.zeros((1,input_dim+1,output_dim))
        self.__X = None
        self.__y = None     
    def train(self,X,y,init_W=np.array([]),rate=0.01,alpha=0,epoch=1000,batch=None,validate_data=None):
        #...
    def update(self,X,y,W,rate,alpha):
        # gradient = - y*(1-yi)*W
        #...
    def predict(self,X):
        #...
    def err_insample(self):
        #...
    def err(self,X,y):
        #...
    def accuarcy_insample(self):
        #...
    def accuarcy(self,X,y):
        #...
    @staticmethod
    def cross_entropy(ys,ys_hat):
        #...
    @staticmethod
    def softmax(zs):
        #...
    def _check_data(self,X,y):
        #...
    def save(self,path):
        #...
    @staticmethod
    def load(path):
        #...
    @staticmethod
    def unit_test():
        #...

def main(): 
    parser = argparse.ArgumentParser(description='HW2: Logistic Regression Training')
    parser.add_argument('-data', metavar='Train_DATA', type=str, nargs='?',
                   help='path of training data',required=True)
    parser.add_argument('-m', metavar='MODEL', type=str, nargs='?',
                   help='path of output model',required=True)
    args = parser.parse_args()

    data = args.data
    model = args.m

    cols = ['data_id','Feature_make','Feature_address','Feature_all','Feature_3d',      
              # ...
           ]
    
    df = pd.read_csv(data,names=cols)
    X = np.array(df.drop(['data_id','label'],axis=1))
    y = np.hstack((np.array(df[['label']]),1-np.array(df[['label']])))
    
    ratio = 0.8
    num_data = X.shape[0]
    num_train = int(ratio * num_data)
    num_valid = num_data - num_train
    
    X_train = X[0:num_train,:]
    y_train = y[0:num_train,:]
    
    X_valid = X[num_train:,:]
    y_valid = y[num_train:,:]

    lg = LogisticRegression(input_dim=57,output_dim=2)
    lg.train(X_train,y_train,rate=7.7e-6,batch=10,epoch=20000,alpha=0,validate_data=(X_valid,y_valid))
    lg.save(model)

if __name__ == "__main__":
    main()

```

Testing at  `logistic_regression_test.py`

**Q2: 使用DNN的方法重複Q1的任務：偵測垃圾信件**

**Ans2:**

使用Keras建立DNN，Training at `dnn.py`

```python
#!python2.7
import sys,os
import numpy as np 
import pandas as pd
import pickle
import math

data = '~/Documents/DeepLearning/NTU_HYLee_MachineLearning/Homework/hw02/spam_data/spam_train.csv'

cols = ['data_id','Feature_make','Feature_address', 
        # ...
       ]

df = pd.read_csv(data,names=cols)
X = np.array(df.drop(['data_id','label'],axis=1))
y = np.hstack((np.array(df[['label']]),1-np.array(df[['label']])))

ratio = 0.8
num_data = X.shape[0]
num_train = int(ratio * num_data)
num_valid = num_data - num_train

X_train = X[0:num_train,:]
y_train = y[0:num_train,:]

X_valid = X[num_train:,:]
y_valid = y[num_train:,:]

import keras
model = keras.models.Sequential()
model.add(keras.layers.Dense(units=10,input_dim=57))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(units=10))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(units=2))
model.add(keras.layers.Activation('softmax'))

adam = keras.optimizers.Adam(lr=0.00004, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy',optimizer=adam, metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=10,epochs=300,validation_data=(X_valid,y_valid))
```

## HW03

Original: [http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/ML%20HW3.pdf](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/ML%20HW3.pdf)

使用影像圖集cifar-10做影像辨識並分類，一共有10個類別，包括：飛機、汽車、鳥、等等，每個類別各有50張已經Label好的圖片，所以Labeled Data有500筆，另外還有45000筆的圖片是Unlabeled Data，只有圖但是不知道它們的類別。

**Q1. Supervised Learning: 實作CNN，使用500筆Labeled Data來作影像辨識和分類。**

**Ans1:**

所有我試過的CNN Model都統一放在`models_supervised_cnn.py`

中間處理過程使用到的資料處理、圖表繪製的工具包，我寫在`common.py` 裡

`supervised_cnn.py` 則是我訓練Supervised CNN的主程式。

訓練完畢的Supervised CNN最好的結果可以到69.4%。

**Q2. Semi-supervised Learning方法一: 使用Self-training的技術將Un-labeled Data包含進來使得Model可以更精準，Self-training的作法是用Labeled Data Training好的CNN去預測Un-labeled Data，並依照信心度去部分認定某些Un-labeled Data為新的Label Data，然後再使用擴增後的Labeled Data更新原本CNN的權重，反覆操作來增進CNN。**

**Ans2:**

在`self_train_cnn.py`中實現了Self-training的技術。

從上面已經訓練好的CNN出發，反覆的用有信心的Un-labeled Data來更新原本的CNN。

在Fitting的過程中發現最重要的是，要讓信心度足夠高的Un-labeled Data才能再做下一輪的更新，否則整個CNN會越訓練越差，因為信心度不夠的Data可能不一定正確，反而增加了雜訊。

訓練完畢的Self-train CNN最好的結果可以到76.2%，的確是有幫助的。

**Q3. Semi-supervised Learning方法二: 利用Autoencoder加上所有Data (Labeled Data + Un-labeled Data) 先做萃取出重要的Features，接下來使用Autoencoder訓練完畢的Encoder轉換Labled Data成新的Features，再做後續CNN的訓練。**

**Ans3:**

在`cnn_autoencoder.py`實現Autoencoder CNN。

訓練完畢的Autoencoder CNN最好的結果可以到67.6%，和Supervised CNN效果差不多而已。

## HW04

Original: [http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/ML%20HW4.pdf](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/ML%20HW4.pdf)

這個部分還沒完成，架構反覆檢查是沒有問題，但是不知道為什麼沒有做出Data分離成群的效果。