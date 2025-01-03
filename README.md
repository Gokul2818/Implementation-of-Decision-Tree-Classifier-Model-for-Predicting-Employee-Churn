# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: GOKUL S
RegisterNumber:  24004336
```
```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/f8e5c5d0-f602-463a-b4da-140a2a26442d)
![image](https://github.com/user-attachments/assets/c8fab2b4-3968-4a5f-a2e0-565fde99f616)
![image](https://github.com/user-attachments/assets/a3beb495-90a1-420b-b10c-0f074ea24dbd)
![image](https://github.com/user-attachments/assets/48e45f6c-12ab-479e-8e9c-d20802196914)
![image](https://github.com/user-attachments/assets/09439818-cd98-415b-8d2d-b076a8b52e9b)
![image](https://github.com/user-attachments/assets/04b71c4c-45c8-48c9-8c01-4af89adf7cce)
![image](https://github.com/user-attachments/assets/54a65e93-0158-450e-8676-b67ad1e36e5e)
![image](https://github.com/user-attachments/assets/9824aa9b-3f55-401d-8bda-01e7a34f3fe2)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
