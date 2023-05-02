# Elastic ML Whitelist Guide

Welcome to the Elastic ML Whitelist Guide! This guide will walk you through creating and using a trained machine learning model to predict whether alerts in Elastic should be whitelisted or not.

![Machine Learning](images/Elastic Rule Exception Machine Learning Model Workflow (3).png)

## Steps

### 1. Train the Model Locally

Train a machine learning model using historical Elastic data. Preprocess data as needed before training the model. Save the trained model to a file (e.g., using `pickle`). As the diagrams shows, we will be 

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
import eland as ed
from elasticsearch import Elasticsearch

```
#### Collecting Data
```python
# Collect to Elastic Search
es = Elasticsearch(
hosts=[http://localhost:9200/],
api_key=('api_id', 'api_key'))

```

```python
# Creating DataFrame from data stored in Elastic Search
ed_df = ed.DataFrame(es, es_index_pattern="Elastic Index Name")
# Showing first 300 rows of data and saving it into a variable
subset_df = ed_df.head(300)
# Converting that Data into a Pandas DataFrame
pd_df = subset_df.to_pandas()
# Storing Data into a CSV
pd_df.to_csv("index.csv", index=False)
```

#### Cleaning of Data

```python
def remove_hyphens_from_csv(file_name):
# Read the CSV file
    df = pd.read_csv(file_name)

# Iterate through each column and remove hyphens
for column in df.columns:
    df[column] = df[column].astype(str).str.replace('-', '')

# Save the modified df to a new CSV file
df.to_csv('clean.csv', index=False)
```
#### Feature Encoding
```python
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
After collecting, cleaning, and feature encoding the data, you want to create a new column entitled "whitelist", and then you want to manually label each row in the dataset under that column either 0 or 1. 1 being whitelist and 0 being do not whitelist. In order to accuratly label these rows correctly you will manually have to analyze and research each alert row by row and decide whether this alert should be whitelisted or not, this will result in the best performance of the model.

#### Model Training
```python
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
# !--Random Forest Model--!

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

```python
# Convert model into a .pkl file

from joblib import dump
dump(model, 'model.pkl')
```

### 3. Import the Model into Elasticsearch

```python
import eland as ed
from elasticsearch import Elasticsearch
import pickle

es_client = Elasticsearch("http://localhost:9200") 

# Replace 'path/to/trained_model.pkl' with the path to saved model file

with open("path/to/trained_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

model_id = "imported-model-id"
ed.ml.import_model(es_client, model, model_id)
```
### OR
```python
from eland.ml import MLModel

>>> es_model = MLModel.import_model(
    es_client="http://localhost:9200",
    model_id="xgb-classifier",
    model=xgb_model,
    feature_names=["f0", "f1", "f2", "f3", "f4"],
)

```

### 4. Create an Elasticsearch Ingest Pipeline with the Inference Processor

```
pipeline_id = "whitelist-prediction-pipeline"

pipeline_body = {
   "description": "Pipeline to predict if an alert should be whitelisted",
   "processors": [
       {
           "script": {
               "lang": "painless",
               "source": """
                 // custom script to apply feature encoding
                 def hash_value(col, value) {
                   return Integer.toUnsignedLong((col + '_' + value).hashCode());
                 }

                 // Apply hashing to each field that has importance
                 ctx['field1_hash'] = hash_value('field1', ctx['field1']);
                 ctx['field2_hash'] = hash_value('field2', ctx['field2']);
                 // ... Add similar lines for all the other fields
               """
           }
       },
       {
           "inference": {
               "model_id": model_id,
               "inference_config": {"classification": {}},
               "field_map": { # Dependant Variables
                   "field1": "field1_hash",
                   "field2": "field2_hash",
                   # ... Map the other fields as needed
               },
               "target_field": "whitelist_prediction", #Independant Variable
           }
       },
   ],
}

es_client.ingest.put_pipeline(id=pipeline_id, body=pipeline_body)
```

### 5. Create a New Pipeline to Route Whitelisted Alerts to a New Index
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

### 6. Configure Filebeat or Logstash to Use the New Pipeline

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

https://www.youtube.com/watch?v=U8fnkzp_sfo

### Terms

##### An Inference Processor: is part of the Elasticsearch ingest pipeline that allows you to use pre-trained Ml Models to enrich data during indexing. So it'll apply the ML Model to incoming documents and add the model's prediction results as new fields in the documents before they are stored in an Elastic index.

##### Feature Encoding: Feature encoding is the process of converting non-numeric data types (e.g., categorical or textual data) into a numerical format that can be used by machine learning algorithms. Many machine learning algorithms require input data to be in a numerical format, so feature encoding is a critical pre-processing step when working with non-numeric features.

##### Accuracy: Accuracy is a metric used to evaluate the performance of a classification model. It is calculated as the ratio of the number of correct predictions to the total number of predictions made. In other words, it measures how well a model correctly classifies instances in the dataset.

##### Precision: Precision is a measure of the accuracy of positive predictions made by a classification model. It is calculated as the ratio of true positive predictions (correctly identified positive instances) to the sum of true positive and false positive predictions (instances incorrectly identified as positive). High precision indicates that a model is good at avoiding false positives.

##### Recall: Recall, also known as sensitivity or true positive rate, is a measure of a classification model's ability to identify all the relevant instances in the dataset. It is calculated as the ratio of true positive predictions to the sum of true positive and false negative predictions (instances incorrectly identified as negative). High recall indicates that a model is good at identifying positive instances and minimizing false negatives.

##### F1 Score:The F1 Score is a metric that combines precision and recall to provide a single measure of a classification model's performance. It is the harmonic mean of precision and recall and ranges between 0 and 1, with 1 indicating perfect precision and recall. The F1 Score is particularly useful when dealing with imbalanced datasets, where one class is more frequent than the other, as it takes into account both false positives and false negatives. A high F1 Score indicates that the model has a good balance between precision and recall.