# Elastic Machine Learning Whitelist Guide

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
Now that we have the features we want our model to be trained with lets concatenate those csv files we previously downloaded (make sure to rename these files accordingly e.g. file1.csv, file2.csv, file3.csv) and then remove every column that is not in our feature list.

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

### Elastic Pipeline

Now that we have our model created we can now create the pipeline that will run our model againts our elastic siem logs. See code below, you can copy and paste the code in the Elastic Dev Console (Make sure to configure Pipeline Name, Description, Mode ID, and Field Map)

```
PUT _ingest/pipeline/ml-mitre-ta0002-execution-pipeline 

{ 

  "description": "Inference pipeline for ml-mitre-ta0002-execution model", 

  "processors": [ 

    { 

      "inference": { 

        "model_id": "ml-mitre-ta0002-execution-model-1686766767712", 

        "inference_config": { 

          "classification": {} 

        }, 

        "field_map": { 

          "user.name": "user_name", 

          "host.name": "host_name", 

          "process.name": "process_name", 

          "process.parent.name": "process_parent_name", 

          "process.executable": "process_executable", 

          "process.parent.executable": "process_parent_executable", 

          "process.working_directory": "process_working_directory", 

          "process.parent.working_directory": "process_parent_working_directory", 

          "process.args": "process_args", 

          "process.parent.args": "process_parent_args", 

          "process.entity_id": "process_entity_id", 

          "process.parent.entity_id": "process_parent_entity_id", 

          "host.os.type": "host_os_type" 

        } 

      } 

    }, 

    { 

      "script": { 

        "source": """ 

          if (ctx.containsKey('user') && ctx.user.containsKey('id') && ctx.user.id instanceof String) { 

            try { 

              ctx.user.id = Long.parseLong(ctx.user.id); 

            } catch (NumberFormatException e) { 

              ctx.user.id = null; 

            } 

          } 

        """ 

      } 

    }, 

    { 

      "set": { 

        "field": "whitelist", 

        "value": "{{ml.inference.whitelist}}" 

      } 

    }, 

    { 

      "script": { 

        "source": """ 

          if (ctx.whitelist.equals("1")) { 

            ctx.whitelist = true; 

          } else if (ctx.whitelist.equals("0")) { 

            ctx.whitelist = false; 

          } 

        """ 

      } 

    } 

  ] 

} 

 
```

### Creating Azure Function

Now that we have the pipeline and the model all done, next we need to set up an Azure Function so that we can automate the process of running our models againts our siem logs. Here is how we do that...

1. Activate your role in azure
2. Download the **"Azure Account", "Azure Functions", and "Azure Resources"** extensions in VSCode.
3. Click on the **Azure Icon** on the left side bar in VSCode.
4. Sign into your **Azure account**.
5. Create an Azure Function by clicking on the **Subscription** of your choice, then right click on **Function App**, then click **Create Function App in Azure**
6. Enter a **Globally Unique** name for you Azure Function.
7. Select the appropriate **Coding Language**
8. Select a **Location**
9. Select a **Resource Group**
10. Now you can insert your code into the .py file called **__init__.py**
