import pandas as pd
from sklearn.metrics import f1_score

file_name1 = r'/Users/yqh/Desktop/label_true.csv'
file_name2 = r'/Users/yqh/Desktop/label_predict.csv'

y_true = pd.read_csv(file_name1, header=0)
y_pred = pd.read_csv(file_name2, header=0)

y11 = pd.read_csv(file_name1, usecols=[0], header=0)
y12 = pd.read_csv(file_name1, usecols=[1], header=0)
y13 = pd.read_csv(file_name1, usecols=[2], header=0)


y21 = pd.read_csv(file_name2, usecols=[0], header=0)
y22 = pd.read_csv(file_name2, usecols=[1], header=0)
y23 = pd.read_csv(file_name2, usecols=[2], header=0)

f1_score(y11, y21, average='micro')
