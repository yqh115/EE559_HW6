from sklearn.svm import LinearSVC
import pandas as pd
from sklearn import preprocessing

file_name1 = r'/Users/yqh/Desktop/norm_training_data.csv'
file_name2 = r'/Users/yqh/Desktop/norm_testing_data.csv'

X_train = pd.read_csv(file_name1, usecols=range(0, 22), header=0)
y1_train = pd.read_csv(file_name1, usecols=[22], header=0)
y2_train = pd.read_csv(file_name1, usecols=[23], header=0)
y3_train = pd.read_csv(file_name1, usecols=[24], header=0)

X_test = pd.read_csv(file_name2, usecols=range(0, 22), header=0)
y1_test = pd.read_csv(file_name2, usecols=[22], header=0)
y2_test = pd.read_csv(file_name2, usecols=[23], header=0)
y3_test = pd.read_csv(file_name2, usecols=[24], header=0)

label_predict = pd.DataFrame(index=range(0, 2159))
label_true = pd.DataFrame(index=range(0, 2159))

y11 = preprocessing.LabelEncoder().fit_transform(y1_test)
y12 = preprocessing.LabelEncoder().fit_transform(y2_test)
y13 = preprocessing.LabelEncoder().fit_transform(y3_test)

label_true.insert(0, 'Family', y11)
label_true.insert(1, 'Genus', y12)
label_true.insert(2, 'Species', y13)

LSVC = LinearSVC(penalty='l1', dual=False, C=2)
LSVC.fit(X_train, y1_train)
y_predict_1 = LSVC.predict(X_test)
LSVC.fit(X_train, y2_train)
y_predict_2 = LSVC.predict(X_test)
LSVC.fit(X_train, y3_train)
y_predict_3 = LSVC.predict(X_test)

y21 = preprocessing.LabelEncoder().fit_transform(y_predict_1)
y22 = preprocessing.LabelEncoder().fit_transform(y_predict_2)
y23 = preprocessing.LabelEncoder().fit_transform(y_predict_3)

label_predict.insert(0, 'Family', y21)
label_predict.insert(1, 'Genus', y22)
label_predict.insert(2, 'Species', y23)

label_true.to_csv('/Users/yqh/Desktop/label_true.csv', index=False, header=label_true.columns)
label_predict.to_csv('/Users/yqh/Desktop/label_predict.csv', index=False, header=label_predict.columns)