from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
import pandas as pd

file_name = r'/Users/yqh/Desktop/norm_training_data.csv'

X = pd.read_csv(file_name, usecols=range(0, 22), header=0)
y = pd.read_csv(file_name, usecols=[22], header=0)
X_resampled, y_resampled = SMOTE().fit_sample(X, y)

model = LinearSVC(penalty='l1', dual=False)
param_grid = {'C': range(1, 20)}
grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=10, n_jobs=1, verbose=1)
grid_search.fit(X_resampled, y_resampled)
best_parameters = grid_search.best_estimator_.get_params()

means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

for para, val in best_parameters.items():
    print(para, val)
