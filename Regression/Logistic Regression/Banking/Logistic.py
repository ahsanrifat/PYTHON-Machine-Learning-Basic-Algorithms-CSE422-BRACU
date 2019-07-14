import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

"""Reading CSV File"""
data = pd.read_csv('banking.csv', header=0)
data = data.dropna()
print(data.shape)
print(list(data.columns))
print()
print()
print("""------Barplot for the dependent variable y(yes/no)------""")
#sns.countplot(x='y',data=data, palette='hls')
#plt.show()
print()
print()
print("""------Consumer job distribution------""")
#sns.countplot(y="job", data=data)
#plt.show()
print()
print()
print("""------Material Status distribution------""")
#sns.countplot(x="marital", data=data)
#plt.show()

"""Our prediction will be based on the customer’s job, marital status, whether he(she) has 
credit in default, whether he(she) has a housing loan, whether he(she) has a personal loan,
and the outcome of the previous marketing campaigns. So, we will drop the variables that
we do not need."""

data.drop(data.columns[[0, 3, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]], axis=1, inplace=True)

"""Data Pre Processing"""


"""Create dummy variables, that is variables with only two values, zero and one.
In logistic regression models, encoding all of the independent variables as dummy 
variables allows easy interpretation and calculation of the odds ratios, and 
increases the stability and significance of the coefficients."""

data2 = pd.get_dummies(data, columns =['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])
#print(list(data2.columns))

"""Drop the unknown columns"""

data2.drop(data2.columns[[12, 16, 18, 21, 24]], axis=1, inplace=True)
#print(data2.columns)

"""Check the independence between the independent variables"""
#sns.heatmap(data2.corr())
#plt.show()