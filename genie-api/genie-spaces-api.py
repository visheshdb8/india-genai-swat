# Databricks notebook source
# MAGIC %md
# MAGIC # Genie Spaces APIs Demo
# MAGIC #### Sample Space: 01ef4ff7110a11709bffa0e18f1a1388
# MAGIC #### Sample Questions:
# MAGIC 1. describe all the tables
# MAGIC 1. What is the weekly average trip distance for taxi trips that occurred in Jan and Feb 2016?
# MAGIC 1. Rank the top 3 drop-off zip codes by average fare amount.
# MAGIC 1. How much total fare was collected from trips that were longer than 10 miles?
# MAGIC 1. Visualize - note that it doesn't create any visuals via api calls but rather produces a response to indicate that we should rather get the result set and use our own visualization tool. Also, in the visual generated by genie from the UI it filters things intelligently (show)

# COMMAND ----------

import requests
import json
import time
from pyspark.sql.types import *
from datetime import datetime
from IPython.display import Markdown

# COMMAND ----------

# dbutils.widgets.text("space_id", "")
# dbutils.widgets.text("chat_name", "")
# dbutils.widgets.text("question", "")""
space_id = dbutils.widgets.get("space_id")
chat_name = dbutils.widgets.get("chat_name")
question = dbutils.widgets.get("question")
token = dbutils.secrets.get("fieldeng", "va-pat-token")
base_url='https://' + spark.conf.get("spark.databricks.workspaceUrl")

max_retries = 10
retry_delay = 10

type_mapping = {
    'STRING': StringType(),
    'INT': IntegerType(),
    'LONG': LongType(),
    'FLOAT': FloatType(),
    'DOUBLE': DoubleType(),
    'SHORT': ShortType(),
    'BYTE': ByteType(),
    'BOOLEAN': BooleanType(),
    'DATE': DateType(),
    'TIMESTAMP': TimestampType(),
    'BINARY': BinaryType(),
    'DECIMAL': DecimalType(10, 0)
}

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create a new Chat
# MAGIC call one time for the demo - no need to call again and again for different related conversations/questions

# COMMAND ----------

url = f"{base_url}/api/2.0/genie/spaces/{space_id}/start-conversation"

payload = json.dumps({
  "content": chat_name
})

headers = {
  'Content-Type': 'application/json',
  'Authorization': f'Bearer {token}'
}

response = requests.request("POST", url, headers=headers, data=payload).json()
conversation_id = response['conversation_id']

# COMMAND ----------

# MAGIC %md
# MAGIC #### Ask a query

# COMMAND ----------

url = f"{base_url}/api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}/messages"

payload = json.dumps({
  "content": question
})

headers = {
  'Content-Type': 'application/json',
  'Authorization': f'Bearer {token}'
}

response = requests.request("POST", url, headers=headers, data=payload).json()
message_id = response['id']

# COMMAND ----------

# MAGIC %md
# MAGIC #### Get results

# COMMAND ----------

url = f"{base_url}/api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}/messages/{message_id}"

payload = ""

headers = {
  'Content-Type': 'application/json',
  'Authorization': f'Bearer {token}'
}

attempt = 0
sql_flag = 0

while attempt < max_retries:
  # print("Check for Genie completion status... attempt: #" + str(attempt + 1))
  response = requests.request("GET", url, headers=headers, data=payload).json()
  # print(response)
  status = response['status']
  
  if status == 'COMPLETED':
    try:
      result = f"SQL: {response['attachments'][0]['query']['query']} \r\n\r\n Description: {response['attachments'][0]['query']['description']}"
      sql_flag = 1
    except:
      result = response['attachments'][0]['text']['content']
    break
  
  elif status != 'COMPLETED' and attempt < max_retries - 1:
    time.sleep(retry_delay)
  
  else:
    print(response)
    raise Exception(f"Query failed or still running after {max_retries*retry_delay} seconds")
    
  attempt += 1

# COMMAND ----------

# MAGIC %md
# MAGIC #### Display results

# COMMAND ----------

if sql_flag ==1:
  url = f"{base_url}/api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}/messages/{message_id}/query-result"
  payload = ""

  headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {token}'
  }

  response = requests.request("GET", url, headers=headers, data=payload).json()

  columns = response['statement_response']['manifest']['schema']['columns']
  rows = []

  for item in response['statement_response']['result']['data_typed_array']:
    row = []
    for i, value in enumerate(item['values']):
      type_name = columns[i]['type_name']

      try:
        if type_name == 'STRING':
          row.append(value['str'])
        elif type_name in {'INT', 'LONG', 'SHORT', 'BYTE'}:
          row.append(int(value['str']))
        elif type_name in {'FLOAT', 'DOUBLE', 'DECIMAL'}:
          row.append(float(value['str']))
        elif type_name == 'BOOLEAN':
          row.append(value['str'].lower() == 'true')
        elif type_name == 'DATE':
          row.append(datetime.strptime(value['str'], '%Y-%m-%d').date())
        elif type_name == 'TIMESTAMP':
          row.append(datetime.strptime(value['str'], '%Y-%m-%dT%H:%M:%S.%fZ'))
        elif type_name == 'BINARY':
          row.append(bytes(value['str'], 'utf-8'))
        else:
          row.append(bytes(value['str'], 'utf-8'))
      except:
        if type_name == 'STRING':
          row.append(value['str'])
        elif type_name in {'INT', 'LONG', 'SHORT', 'BYTE'}:
          row.append(int(value['str']))
        elif type_name in {'FLOAT', 'DOUBLE', 'DECIMAL'}:
          row.append(float(value['str']))
        elif type_name == 'BOOLEAN':
          row.append(value['str'].lower() == 'true')
        elif type_name == 'DATE':
          row.append(datetime.strptime(value['str'], '%Y-%m-%d').date())
        elif type_name == 'TIMESTAMP':
          row.append(datetime.strptime(value['str'], '%Y-%m-%d %H:%M:%S'))
        elif type_name == 'BINARY':
          row.append(bytes(value['str'], 'utf-8'))
        else:
          row.append(bytes(value['str'], 'utf-8'))
          
    rows.append(row)
    
  # schema = StructType([StructField(col, StringType(), True) for col in columns])
  schema = StructType([StructField(col['name'], type_mapping[col['type_name']], True) for col in columns])
  df = spark.createDataFrame(rows, schema)
  display(df)

else:
  print(result)

# COMMAND ----------

Markdown(f"[Genie Space Link]({base_url}/genie/rooms/{space_id})")