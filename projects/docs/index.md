# Elastic ML Whitelist Guide

Welcome to the Elastic Machine Learning Whitelist Guide! This guide will walk you through creating and using a trained machine learning model to predict whether alerts in Elastic should be whitelisted or not.

![Machine Learning](images/Elastic Rule Exception Machine Learning Model Workflow (3).png)

## Steps

### Get the data

The first step to training a machine learning model is to get the data that you need, for this specific project we will download the data that we need from Elastic. We can do that by going to **Elastic> Click on the hamburger icon at the top left of the screen> Under the Analytics section click on Discover.**

Now that we are on the Discover page we need to switch our index to our siem index after this, we can then begin to download the necessary data, first lets select a date by clicking on the time picker then, we can click on **Share** at the top right of the screen, and click **CSV Reports> Generate CSV**. We want to download as much data as we need and we can do so by repeating the steps above as there is a limit on how much data you can download at once.

### Getting the data ready: Transform & Clean

Now that we have our data we want to get our data ready for training, we are going to need to concatenate all those csv files that we previously downloaded and then we want to clean up the data a bit, here is how we do that.

This code will request the list of features you would like your model to have and, put those features in parenthesis and add a comma at the end of each feature so that you can easily copy and paste the text into the code below.
```python

words_input = input("Please paste the list of words: ")

words = words_input.split()

processed_words = "\n".join([f"'{word}'," for word in words])

print(processed_words)

```
Now that we have the features we want our model to be trained with lets concatenate those csv files we previously downloaded and then remove every column that is not in our feature lits.

```python
import pandas as pd
import numpy as np

files = ['file1.csv', 'file2.csv', 'file3.csv', 'file4.csv', 'file5.csv', 'file6.csv', 'file7.csv', 'file8.csv']

dfs = []
for f in files:
    df = pd.read_csv(f, on_bad_lines='warn')
    print(f"File: {f}, Columns: {df.columns.tolist()}")
    dfs.append(df)

result = pd.concat(dfs, ignore_index=True)

cols_to_keep = [
    'kibana.alert.rule.name',
    'user.name',
    'user.domain',
    'host.name',
    'process.name',
    'event.category',
    'source.ip',
    'source.port',
    'destination.port',
    'dns.question.name',
    'dns.question.type',
    'file.name',
    'file.path',
    'dll.name',
    'dll.path',
    'process.parent.name',
    'process.executable',
    'process.working_directory',
    'process.args',
    'process.hash.sha256',
    'dll.hash.sha256',
    'signal.reason'
]

cols_to_keep = [col for col in cols_to_keep if col in result.columns]

result = result[cols_to_keep]

result = result.replace('-', np.nan)

result.to_csv('siem.csv', index=False, header=True)
```

### Elastic Model Trainng
To import our data into elastic **Click on the hamburger icon at the top left of the screen> Then on Machine Learning> and then on File> Import the appropriate csv file that was concatenated and cleaned.** Now that we have our data ready to go, lets train our model. **Click on the hamburger icon at the top left of the screen> Under Analytics click on Machine Learning> Under Data Frame Analytics click on Jobs> Then click Create job** As we create our ML mode make sure to set "Feature Importance Values to the amount of features we have.