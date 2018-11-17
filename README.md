# NTU H.Y. Lee Machine Learning Homework

![image](https://img.shields.io/badge/python-2.7-blue.svg)

## Prepare

### virtual environment

create virtual environment

```
$ virtualenv ./ENV
```

enter virtual environment

```
$ source ./ENV/bin/active
```

if you want to exit virual environment, 

```
$ deactivate
```

### install dependencies

install dependencies under virtual environment

```
$ pip2.7 install -r requirements.txt
```

## HW00

Original: [https://docs.google.com/presentation/d/1VCnqWX469V4Qi_dHJIXxWMhvtg3h6pyBuRYPao-GZ1k/edit#slide=id.g17e2c04840_1_14](https://docs.google.com/presentation/d/1VCnqWX469V4Qi_dHJIXxWMhvtg3h6pyBuRYPao-GZ1k/edit#slide=id.g17e2c04840_1_14)

**Q1. File `given/hw0_data.dat` has 11 columns splitted by single space. Please choose specific column `i` from this file, sort the sequence from small to large, and output to `result/ans1.txt`. Given `i = 1`.**

```
$ cd hw00
$ ./Q1.sh 1 given/hw0_data.dat
```

**Q2. Input the picture `given/Lena.png`. Please let this picture upside down and then left/right reversed (rotate 180 degree). Output to `result/ans2.png`**

```
$ cd hw00
$ ./Q2.sh given/Lena.png
```

## HW01

Original: [http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/hw1.pdf](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/hw1.pdf)

data set: This data is an observation from 豐原 station, recorded weather parameters each hour at one day.
* `given/train.csv`: Choose first 20 days in each month to be a training set.
* `given/test.csv`: Choose last 10 days in each month to be a testing set. From testing set, select data among continuous 10 hours as a batch. Use first 9 hours data in this batch as a feature and `PM2.5` at 10 hour as answer. And we hide this answer.

Please train a linear model to predict the answer in `given/test.csv` (format of output reference `given/sampleSubmission.csv`).

**Ans:**

Here presents two methods to implement linear regresssion.

1. using pseudo inverse,

```
$ cd hw01
$ python main.py --method pseudo_inverse --output result/pseudo_inverse.csv
```

2. using gradient descent,

```
$ cd hw01
$ python main.py --method gradient_descent --output result/gradient_descent.csv
````

## HW02

Orignal: [http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/hw2.pdf](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/hw2.pdf)

**Q1: Implement logistic regression to detect spams**

We have some labeled emails. In `given/spam_train.csv`, first severval columns present features of words and the last column presents spam/not spam labels. Those features are explained in `given/spambase.names`. Please implement logistic regression to predict whether letters are spams or not in `given/spam_test.csv`?

**Ans1:**

train:

```
$ cd hw02
$ python logistic_regression.py --type train --model result/model1.p
```

test:

```
$ python logistic_regression.py --type test --model result/model1.p --output result/result1.csv
```

**Q2: Implement DNN to detect spams**

**Ans2:**

train:

```
$ cd hw02
$ python dnn.py --type train --model result/model2.h5
```

test:

```
$ python dnn.py --type test --model result/model2.h5 --output result/result2.csv
```

## HW03

Original: [http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/ML%20HW3.pdf](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/ML%20HW3.pdf)

please prepare dataset of cifar-10 first
```
$ cd hw03
$ python prepare_data.py
```

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
