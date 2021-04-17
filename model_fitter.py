import pickle
import pandas as pd

from utils.data_preprocessor import DataLoader
from settings.constants import TRAIN_CSV
from settings.constants import TEST_CSV
from utils.grid_search_est import GridSearchEstimator

# read data
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

# preprocess train data before fit operation
train_dataloader = DataLoader(train)
X_train, y_train = train_dataloader.preprocess()

# preprocess test data
test_dataloader = DataLoader(test)
X_test, y_test = test_dataloader.preprocess()

print(X_train.head())

# call GridSearchEstimator class in order to find best estimator
estimator = GridSearchEstimator(X_train, y_train)

# define GridSearch best_parameters
grid_params = dict(n_estimators=[100, 300])  # RF hyperparameters
cv = 4  # number of cross validation runs
scoring = 'f1'  # scoring metrics

best_score, best_estimator, best_params = estimator.rfc_grid_search(grid_params, cv, scoring)

# dump best estimator to GridSearch.pickle file
with open('models/GridSearch.pickle', 'wb')as f:
    pickle.dump(best_estimator, f)

print("Best GridSearch f1_score:", best_score)
print("Best estimator params:", best_params)
