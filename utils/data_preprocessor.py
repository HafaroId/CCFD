class DataLoader(object):
    """
    A class used to preprocess data before modeling ir predicting operations
    ...
    Attributes
    __________
    dataset : pandas DataFrame
    Methods
    -------
    preprocess()
        split dataset to X and y
        preprocess data before modeling or predicting
    """
    def __init__(self, dataset):
        self.dataset = dataset.copy()

    def preprocess(self):
        y = self.dataset["Class"]  # target assignment
        self.dataset.drop(["Class", "Time"], axis=1, inplace=True)  # drop target and unuseful "Time" column
        X = self.dataset
        return X, y
