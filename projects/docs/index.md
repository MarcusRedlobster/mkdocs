# Elastic ML Whitelist Guide

Welcome to the Elastic ML Whitelist Guide! This guide will walk you through creating and using a trained machine learning model to predict whether alerts in Elastic should be whitelisted or not.

![Machine Learning](images/Elastic Rule Exception Machine Learning Model Workflow (3).png)

## Steps

### 1. Train the Model Locally

Train a machine learning model using historical Elastic data. Preprocess data as needed before training the model. Save the trained model to a file (e.g., using `pickle`).

### 2. Import the Model into Elasticsearch

```python
import eland as ed
from elasticsearch import Elasticsearch
import pickle

es_client = Elasticsearch("http://localhost:9200") 

```

Replace 'path/to/trained_model.pkl' with the path to saved model file

```
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