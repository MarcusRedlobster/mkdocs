# Elastic ML Whitelist Guide

Welcome to the Elastic ML Whitelist Guide! This guide will walk you through creating and using a trained machine learning model to predict whether alerts in Elastic should be whitelisted or not.

![Machine Learning](images/Elastic Rule Exception Machine Learning Model Workflow (3).png)

## Steps

### 1. Train the Model Locally

Train a machine learning model using historical Elastic data. Preprocess data as needed before training the model. Save the trained model to a file (e.g., using `pickle`).

#### Imports
```python
import pandas as pd
import numpy as np 
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import hashlib
from sklearn.inspection import permutation_importance

```

#### Cleaning of Data

```python

# !--CLEANING OF data--!

def remove_hyphens_from_csv(file_name):
# Read the CSV file
    train = pd.read_csv(file_name)

# Iterate through each column and remove hyphens
for column in train.columns:
    train[column] = train[column].astype(str).str.replace('-', '')

# Save the modified train to a new CSV file
train.to_csv('clean.csv', index=False)

```

```python
# !--Feature Encoding--!

# Read data
df = pd.read_csv('clean.csv')

# function that takes a row and hashes each value with the column name
def hash_row_values(row):
    return [hash(f"{col}_{value}") for col, value in zip(row.index, row) if col != 'whitelist']

# Apply the hash function to each column
for col in df.columns:
    if col != 'whitelist':
        df[col + '_hash'] = df.apply(lambda row: hash(f"{col}_{row[col]}"), axis=1)

```

```python
# !--Model Training--!

# Independant Variables
X = df[['Features']]

#Dependant Variable
y = df[['What is being predicted']]
```
```python
# Train Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
```python
from sklearn.ensemble import RandomForestClassifier

Rmodel = RandomForestClassifier(n_estimators=100)

Rmodel.fit(X_train, y_train)

y_pred = Rmodel.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

Accuracy: 0.9726027397260274
Precision: 0.9523809523809523
Recall: 0.9523809523809523
F1 Score: 0.9523809523809523
```

### 2. Import the Model into Elasticsearch

```python
import eland as ed
from elasticsearch import Elasticsearch
import pickle

es_client = Elasticsearch("http://localhost:9200") 

```

```python
# Replace 'path/to/trained_model.pkl' with the path to saved model file

with open("path/to/trained_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

model_id = "imported-model-id"
ed.ml.import_model(es_client, model, model_id)

```

### 3. Create an Elasticsearch Ingest Pipeline with the Inference Processor

```
pipeline_id = "whitelist-prediction-pipeline"

pipeline_body = {
    "description": "Pipeline to predict if an alert should be whitelisted",
    "processors": [ 
        {
            "inference": {
                "model_id": model_id,
                "inference_config": {"classification": {}},
                "field_map": {}, # Dependant Variables
                "target_field": "whitelist_prediction", #Independant Variable
            }
        }
    ],
}

es_client.ingest.put_pipeline(id=pipeline_id, body=pipeline_body)

```

### 4. Create a New Pipeline to Route Whitelisted Alerts to a New Index
```
whitelisted_alerts_index = "whitelisted-alerts"
new_pipeline_id = "route-to-whitelisted-index"

new_pipeline_body = {
    "description": "Pipeline to route whitelisted alerts to a new index",
    "processors": [
        {
            "conditional": {
                "if": "ctx.whitelist_prediction.class_name == '1'",
                "processors": [
                    {
                        "index": {
                            "index": whitelisted_alerts_index,
                        }
                    }
                ],
            }
        },
    ],
}

es_client.ingest.put_pipeline(id=new_pipeline_id, body=new_pipeline_body)

```

### 5. Configure Filebeat or Logstash to Use the New Pipeline

# filebeat.yml
```
filebeat.inputs:
- type: log
  paths:
    - /path/to/alert/logs/*.log
  fields:
    pipeline: "route-to-whitelisted-index"

output.elasticsearch:
  hosts: ["http://localhost:9200"]
  pipeline: "%{[fields.pipeline]}"
```
# logstash.conf
```
output {
  elasticsearch {
    hosts => ["http://localhost:9200"]
    index => "siem-index"
    pipeline => "route-to-whitelisted-index"
  }
}
```

### Reference Links

https://eland.readthedocs.io/en/v8.3.0/reference/api/eland.ml.MLModel.import_model.html

https://eland.readthedocs.io/en/7.9.1a1/examples/introduction_to_eland_webinar.html#Machine-Learning-Demo

https://www.elastic.co/guide/en/elasticsearch/reference/current/ingest.html

https://www.elastic.co/guide/en/elasticsearch/client/eland/current/overview.html

https://www.youtube.com/watch?v=w8RwRO8gI_s&pp=ygUNZWxhbmQgZWxhc3RpYw%3D%3D

### Index

An Inference Processor is part of the Elasticsearch ingest pipeline that allows you to use pre-trained Ml Models to enrich data during indexing. So it'll apply the ML Model to incoming documents and add the model's prediction results as new fields in the documents before they are stored in an Elastic index.