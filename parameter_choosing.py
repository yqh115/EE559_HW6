from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pandas as pd
import numpy as np

Folder_Path = r'/Users/yqh/Desktop/'
File_name = r'training_data.csv'

b=[]
a = pd.read_csv(Folder_Path + 'norm_training_data.csv', usecols=range(0, 22), header=0)
b = pd.read_csv(Folder_Path + 'norm_training_data.csv', usecols=[22], header=0)

model = SVC(kernel='rbf', probability=True)
param_grid = {'C': range(5, 15), 'gamma': np.arange(0.1, 2, 0.1)}
grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=10, n_jobs=1, verbose=1)
grid_search.fit(a, b)
best_parameters = grid_search.best_estimator_.get_params()

means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

for para, val in best_parameters.items():
    print(para, val)
# model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
# model.fit(a, b)



