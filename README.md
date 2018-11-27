# NTU H.Y. Lee Machine Learning Homework

![image](https://img.shields.io/badge/python-2.7-blue.svg)

## Prepare

### virtual environment and dependencies

#### method 1: virtualenv
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


install dependencies under virtual environment

```
$ pip2.7 install -r requirements.txt
```


#### method 2: pyenv + pipenv

use `pyenv` to local python version to this project,

```
$ pyenv install 2.7.15
$ pyenv local
```

use `pipenv` to set up dependencies,

```
$ pipenv install
```

enter virtual environment

```
$ pipenv shell
```

### GPU support on tensorflow

please install NVIDIA graphics driver, CUDA 9.0 and cuDNN 7.0 first. (ref: [https://medium.com/@zhanwenchen/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e](https://medium.com/@zhanwenchen/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e) )  

otherwise, please choose cpu version tensorflow in `requirements.txt`


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



The `CIFAR-10` dataset (https://www.cs.toronto.edu/~kriz/cifar.html) consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The 10 classes include airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck, labeled 0-9 in order.

In this problem, we picked 500 images in each class from training set as labeled data, and hidden the other 45000 images' label from training set as unlabeled data.

Please use below code to prepare data as above instruction.

```
$ cd hw03
$ python prepare_data.py
```

* `given/all_label.p` contains ten classes (0-9), each class has 500 images.
* `given/all_unlabel.p` contains 45000 images
* `given/test.p` contains 10000 images
* `given/test_ans.txt` are labels of `given/test.p`

**Q1. Supervised Learning: Use `given/all_label.p` data and CNN to predict `given/test.p`**

**Ans1:**

```
$ cd hw03
$ python supervised_cnn.py --type train --model_config ycnet3 --model_name 002
```

The best validation accuracy can reach 69.4%.

**Q2. Semi-supervised Learning Method 1: Self-training method. Try to use trained supervised model to label unlabeled data above specific reliablity threshold. Add those trusted data into labeled data and then use the augmented data to update CNN model.**

**Ans2:**

```
$ cd hw03
$ python self_train_cnn.py --type train --model_config ycnet3 --model_name 002
```

Go from above Q1 trained supervised CNN model and use unlabeled data to update it.

In my observation, the key point is that reliablity threshold must be high enough. Otherwise the CNN model would get worst because adding incorrect labeled data causes more noise.

The best validation accuracy can reach 76.2%. Self-training method is work.


**Q3. Semi-supervised Learning Method 2: Use all data (labeled data + unlabeled data) to pre-train autoencoder and extract some features of data. And use encoder in this autoencoder to do supervised learning on labeled data.**

**Ans3:**

```
$ cd hw03
$ python cnn_autoencoder.py --type train --model_config AutoencoderClassifier01 --model_name 001
```

The best validation accuracy can reach 67.6%. It is at same level with supervised CNN.

## HW04

Original: [http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/ML%20HW4.pdf](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/ML%20HW4.pdf)

這個部分還沒完成，架構反覆檢查是沒有問題，但是不知道為什麼沒有做出Data分離成群的效果。
