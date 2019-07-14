import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import math
import xlrd


"""-------PART 01 --- Collecting Data-------"""



data_titanic=pd.read_excel("titanic3.xls")
print("""Making List of all features""")
features_list=list(data_titanic.columns)
#print(features_list)
# """Number of passengers"""
# print("# of passengers in original data: "+str(len(data_titanic.index)))


"""------PART02-Analyzing DATA------"""


# print("""Count plot(Seaborn)-Survived""")
# sns.countplot(x="survived",data=data_titanic)
# plt.show()
# print("""How many male and female survived""")
# sns.countplot(x="survived",hue='sex',data=data_titanic)
# plt.show()
# print("""Survival based on passengers""")
# sns.countplot(x="survived",hue='pclass',data=data_titanic)
# plt.show()
# print("""Histogrm according to ages""")
# data_titanic['age'].plot.hist()
# plt.show()
# print("Histogram according to fair")
# data_titanic['fare'].plot.hist(bins=20, figsize=(10,5))
# plt.show()
# print("Number of sibling and spouce(sibsp)")
# sns.countplot(x='sibsp',data=data_titanic)
# plt.show()
# print("Number of parch-parents and childern")
# sns.countplot(x='parch',data=data_titanic)
# plt.show()
#print(data_titanic.shape)

"""------PART 03->Data Wrangling(Cleaning your data)------"""


"""Removing null values"""

#print("""Checking if there is any null value""")
#print(data_titanic.isnull())
# print("""Sum of which column has how many null values""")
# print(data_titanic.isnull().sum())
# print("How many nulls using heat map")
# sns.heatmap(data_titanic.isnull(),yticklabels=False,cmap="viridis")
# plt.show()
# print("pClass having age")
# sns.boxenplot(x='pclass',y='age',data=data_titanic)
# plt.show()

"""Drop Cabin as it has many null values"""
data_titanic.drop('cabin',axis=1,inplace=True)
#print(data_titanic.columns)
"""Drop All NAn Values"""
data_titanic.dropna(how='any',axis=0)
# print("After Droping Null Values-(dataframe.dropna)")
# print(data_titanic.isnull().sum())
print("Making values Catagorical")
sex=pd.get_dummies(data_titanic['sex'],drop_first=True)
# print(sex.head(5))
# print()
embark=pd.get_dummies(data_titanic['embarked'],drop_first=True)
# print(embark.head(5))
# print()
pcl=pd.get_dummies(data_titanic['pclass'],drop_first=True)
# print(pcl.head(5))
"""Now have to concat all these columns"""
data_titanic=pd.concat([data_titanic,sex,embark,pcl],axis=1)
"""Dropping the unnecessary columns"""
data_titanic.drop(['sex','embarked','pclass','embarked','name','ticket'],axis=1,inplace=True)
#print(data_titanic.shape)



"""------Part 04-> Split the data intotrain and test"""


X=data_titanic.drop("survived",axis=1)
Y=data_titanic['survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

logistic=LogisticRegression()
logistic.fit(X_train,Y_train)
#
# prediction=logistic.predict(X_test)
# classification_report(Y_test,prediction)

# prediction=logistic.predict(X_test)
# classification_report(Y_test,prediction)
#
# print(confusion_matrix(Y_test,prediction))



