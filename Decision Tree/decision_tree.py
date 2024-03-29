import pandas as pd
from pandas import read_csv
from sklearn import tree


data = read_csv("data.csv")

data['Color'] = data['Color'].map({'Red': 0, 'Blue': 1})
data['Brand'] = data['Brand'].map({'Snickers': 0, 'Kit Kat': 1})

predictors = ['Color', 'Brand']

X = data[predictors]
Y = data.Class

decisionTreeClassifier = tree.DecisionTreeClassifier(criterion="entropy")
dTree = decisionTreeClassifier.fit(X, Y)

dotData = tree.export_graphviz(dTree, out_file=None)
print(dotData)

data['Taste']=pd.Series(['sweet','sweet','sweet','sweet','sweet','sweet','sweet','sweet','sweet','sweet','bitter','bitter','bitter','bitter','bitter','bitter','bitter','bitter','sweet','sweet','sweet','sweet','sweet','sweet','sweet','sweet','sweet','sweet','bitter','bitter','bitter','bitter','bitter','bitter','bitter','bitter','sweet','sweet','sweet','sweet','sweet','bitter','bitter','bitter','bitter','bitter','bitter','bitter','sweet','sweet'])
print(data)

data['Taste'] = data['Taste'].map({'sweet': 0, 'bitter': 1})

predictors = ['Color', 'Brand','Taste']

X = data[predictors]
Y = data.Class

decisionTreeClassifier = tree.DecisionTreeClassifier(criterion="entropy")
dTree = decisionTreeClassifier.fit(X, Y)

dotData = tree.export_graphviz(dTree, out_file=None)
print(dotData)