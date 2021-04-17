import json
import requests
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score

from settings.constants import TEST_CSV
from utils.data_preprocessor import DataLoader

# read test dataset
test = pd.read_csv(TEST_CSV, header=0)

# preprocess test dataset for prediction operation
dataloader = DataLoader(test)
test_x, test_y = dataloader.preprocess()

# pass test data to the API
req_data = {'data': json.dumps(test_x.to_dict())}
response = requests.get('http://127.0.0.1:8000/predict', data=req_data)
api_predict = response.json()['prediction']

print('predict: ', api_predict[:10])
print("___________________________________")
print("accuracy_score:", accuracy_score(api_predict, test_y))
print("roc_auc_score:", roc_auc_score(api_predict, test_y))
print("f1_score:", f1_score(api_predict, test_y))
print(confusion_matrix(api_predict, test_y))
print("___________________________________")
