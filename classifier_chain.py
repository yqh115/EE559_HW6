import numpy as np
import pandas as pd
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import jaccard_similarity_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.svm import LinearSVC


file_name1 = r'/Users/yqh/Desktop/norm_training_data.csv'
file_name2 = r'/Users/yqh/Desktop/norm_testing_data.csv'

X_train = pd.read_csv(file_name1, usecols=range(0, 22), header=0)
y_train1 = pd.read_csv(file_name1, usecols=[22], header=0)
y_train2 = pd.read_csv(file_name1, usecols=[23], header=0)
y_train3 = pd.read_csv(file_name1, usecols=[24], header=0)
y11 = preprocessing.LabelEncoder().fit_transform(y_train1)
y12 = preprocessing.LabelEncoder().fit_transform(y_train2)
y13 = preprocessing.LabelEncoder().fit_transform(y_train3)
y11 = pd.DataFrame(y11, columns=['Family'])
y12 = pd.DataFrame(y12, columns=['Genus'])
y13 = pd.DataFrame(y13, columns=['Species'])


X_test = pd.read_csv(file_name2, usecols=range(0, 22), header=0)
y_test1 = pd.read_csv(file_name2, usecols=[22], header=0)
y_test2 = pd.read_csv(file_name2, usecols=[23], header=0)
y_test3 = pd.read_csv(file_name2, usecols=[24], header=0)
y21 = preprocessing.LabelEncoder().fit_transform(y_test1)
y22 = preprocessing.LabelEncoder().fit_transform(y_test2)
y23 = preprocessing.LabelEncoder().fit_transform(y_test3)
y21 = pd.DataFrame(y21, columns=['Family'])
y22 = pd.DataFrame(y22, columns=['Genus'])
y23 = pd.DataFrame(y23, columns=['Species'])

y_train = pd.DataFrame(index=range(0, 5036))
y_train.insert(0, 'Species', y13)
# y_train.insert(1, 'Genus', y12)
# y_train.insert(2, 'Species', y13)
X_train.insert(22, 'Family', y11)
X_train.insert(23, 'Genus', y12)


y_test = pd.DataFrame(index=range(0, 2159))
y_test.insert(0, 'Species', y23)
# y_test.insert(1, 'Genus', y22)
# y_test.insert(2, 'Species', y23)
X_test.insert(22, 'Family', y21)
X_test.insert(23, 'Genus', y22)

# Fit an independent logistic regression model for each class using the
# OneVsRestClassifier wrapper.
ovr = OneVsRestClassifier(LinearSVC())
ovr.fit(X_train, y_train)
y_pred_ovr = ovr.predict(X_test)
ovr_jaccard_score = jaccard_similarity_score(y_test, y_pred_ovr)

# Fit an ensemble of logistic regression classifier chains and take the
# take the average prediction of all the chains.
chains = [ClassifierChain(LogisticRegression(), order='random', cv=10, random_state=None)]#,for i in range(4)]
for chain in chains:
    chain.fit(X_train, y_train)

y_pred_chains = np.array([chain.predict(X_test) for chain in
                          chains])
chain_jaccard_scores = [jaccard_similarity_score(y_test, y_pred_chain >= .5)
                        for y_pred_chain in y_pred_chains]

y_pred_ensemble = y_pred_chains.mean(axis=0)
ensemble_jaccard_score = jaccard_similarity_score(y_test, y_pred_ensemble >= .5)

model_scores = [ovr_jaccard_score] + chain_jaccard_scores
model_scores.append(ensemble_jaccard_score)

model_names = ('Independent', 'Ensemble')

x_pos = np.arange(len(model_names))

print(model_scores)
print(ovr_jaccard_score)
print(ensemble_jaccard_score)