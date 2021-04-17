from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class GridSearchEstimator:
    """
    A class used to construct GridSearch
    ...
    Attributes
    __________
    X_train : pandas DataFrame
        train dataset
    y_train : pandas DataFrame
        target dataset
    Methods
    -------
    rfc_grid_search()
        construct Pipelines based on RandomForestClassifier
    """
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def rfc_grid_search(self, grid_params=None, cv=4, scoring='f1'):
        if grid_params is None:
            grid_params = dict(n_estimators=[100, 300, 500])

        grid_search = GridSearchCV(RandomForestClassifier(),
                                   param_grid=grid_params,
                                   scoring=scoring, cv=cv)
        grid_search.fit(self.X_train, self.y_train)

        return grid_search.best_score_, grid_search.best_estimator_, grid_search.best_params_
