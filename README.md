# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.

## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: KISHORE.S
RegisterNumber:  212222240050

import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
1. Result output
2. data.head()
3. data.info()
4. data.isnull().sum()
5. Y_prediction value
6. Accuracy value

## Output:

##  Result output :
![image](https://github.com/Kishore2o/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118679883/981f97b6-933c-41dd-8d66-dfe41d47db99)
## data.head() :
![image](https://github.com/Kishore2o/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118679883/47661043-c61f-42ea-8145-51a263772541)
## data.info() :
![image](https://github.com/Kishore2o/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118679883/c94d905d-1926-4a81-bda4-e4dfc385e660)
## data.isnull().sum() :
![image](https://github.com/Kishore2o/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118679883/8bca0928-7082-4377-80b8-b51f3e824c0b)
## Y_prediction value :
![image](https://github.com/Kishore2o/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118679883/012f72b6-9fe7-429d-8088-d4f60b56578f)
## Accuracy value :
![image](https://github.com/Kishore2o/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118679883/c5f3a106-7eff-4d06-b4b8-40e887d64a52)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
