# Databricks notebook source
# MAGIC %md
# MAGIC ####Install Libraries
# MAGIC
# MAGIC for GPU clusters (tested on 14.3 LTS ML GPU with single node g5.2xlarge), use:
# MAGIC ```%pip install -U gradio==4.31.4 boto3==1.34.108 langchain==0.2.0 langchain-community==0.2.0 pypdf==4.2.0 sentence-transformers==2.7.0 faiss-gpu-cu11==1.8.0.2 mlflow==2.11.0 -q```
# MAGIC
# MAGIC for CPU clusters (tested on 14.3 LTS DBR Non-ML with single node r5.xlarge), use:
# MAGIC ```%pip install -U gradio==4.44.0 boto3==1.35.26 langchain==0.3.0 langchain-community==0.3.0 pypdf==5.0.0 sentence-transformers==3.1.1 langchain-huggingface==0.1.0 faiss-cpu==1.8.0.post1 mlflow==2.16.2 -q```
# MAGIC
# MAGIC ####Credentials
# MAGIC
# MAGIC update the line
# MAGIC ```token = dbutils.secrets.get(scope='db-field-eng', key='va-pat-token')``` in cell 7 to either use your PAT token directly or the appropriate scope and key values

# COMMAND ----------

# MAGIC %md
# MAGIC ####Install Libraries

# COMMAND ----------

# %pip install -U gradio==4.31.4 boto3==1.34.108 langchain==0.2.0 langchain-community==0.2.0 pypdf==4.2.0 sentence-transformers==2.7.0 faiss-gpu-cu11==1.8.0.2 mlflow==2.11.0 -q

%pip install -U gradio==4.44.0 boto3==1.35.26 langchain==0.3.0 langchain-community==0.3.0 pypdf==5.0.0 sentence-transformers==3.1.1 langchain-huggingface==0.1.0 faiss-cpu==1.8.0.post1 mlflow==2.16.2 -q
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Import Libraries

# COMMAND ----------

import gradio as gr
import os
import aiofiles
import asyncio
import requests
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatDatabricks
from langchain.chains import RetrievalQA

# COMMAND ----------

# MAGIC %md
# MAGIC ####Define Helper Functions

# COMMAND ----------

def create_volume(catalog, schema, volume):
  
  output_messages = []
  workspaceUrl = 'https://' + spark.conf.get("spark.databricks.workspaceUrl")
  # token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get() # Not recommended and can be replaced your your own user token
  token = dbutils.secrets.get(scope='db-field-eng', key='va-pat-token')
  
  headers = {
    'Content-Type': 'application/json',
    'Authorization': f"Bearer {token}"
  }

  #Create Catalog
  url = f"{workspaceUrl}/api/2.1/unity-catalog/catalogs"
  payload = json.dumps({"name": catalog})
  response = requests.request("POST", url, headers=headers, data=payload)
  if response.status_code==200:
    output_messages.append(f"Catalog {catalog} created successfully ✅")
  else:
    output_messages.append(response.json()['message'])

  #Create Schema
  url = f"{workspaceUrl}/api/2.1/unity-catalog/schemas"
  payload = json.dumps({
    "name": schema,
    "catalog_name": catalog
    })
  response = requests.request("POST", url, headers=headers, data=payload)
  if response.status_code==200:
    output_messages.append(f"Schema {schema} created successfully ✅")
  else:
    output_messages.append(response.json()['message'])
  
  #Create Volume
  url = f"{workspaceUrl}/api/2.1/unity-catalog/volumes"
  payload = json.dumps({
    "catalog_name": catalog,
    "schema_name": schema,
    "name": volume,
    "volume_type": "MANAGED"
  })
  response = requests.request("POST", url, headers=headers, data=payload)
  if response.status_code==200:
    output_messages.append(f"Volume {volume} created successfully ✅")
  else:
    output_messages.append(response.json()['message'])
    
  return "\n".join(output_messages)

# COMMAND ----------

def create_embeddings(uploaded_files, parent_dir):
  
  output_messages = []
  all_pages=[]

  for uploaded_file in uploaded_files:
    filepath = os.path.join(parent_dir, os.path.basename(uploaded_file))
    loader = PyPDFLoader(filepath)
    pages = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200))
    all_pages.extend(pages)
    output_messages.append(f"Embeddings created successfully for {uploaded_file.split('/')[-1]}")

  embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
  db = FAISS.from_documents(all_pages, embedding_function)
  db.save_local(f"{parent_dir}/llama")
  
  return "\n".join(output_messages)

# COMMAND ----------

async def copy_file(src, dst):
  async with aiofiles.open(src, 'rb') as fsrc, aiofiles.open(dst, 'wb') as fdst:
    while True:
      buffer = await fsrc.read(10*1024*1024)  # Read in 1MB chunks
      if not buffer:
        break
      await fdst.write(buffer)

async def upload_file_async(uploaded_files, catalog, schema, volume):
  output_messages = []
  vol_check_msg = create_volume(catalog, schema, volume)
  output_messages.append(vol_check_msg)
  destination_dir = os.path.join(os.path.join(os.path.join("/Volumes", catalog), schema), volume)
  
  tasks = []
  for uploaded_file in uploaded_files:
    try:
      destination_path = os.path.join(destination_dir, os.path.basename(uploaded_file))
      tasks.append(copy_file(uploaded_file, destination_path))
      output_messages.append(f"File '{os.path.basename(uploaded_file)}' is being uploaded to {destination_path}")
    except Exception as e:
      output_messages.append(f"Error with file upload '{os.path.basename(uploaded_file)}': {str(e)}")
  
  # Await all file copy operations
  await asyncio.gather(*tasks)

  try:
    emb_check_msg = create_embeddings([os.path.join(destination_dir, os.path.basename(f)) for f in uploaded_files], destination_dir)
    output_messages.append(emb_check_msg)
  except Exception as e:
    output_messages.append(f"Error with embedding files: {str(e)}")
  
  return "\n".join(output_messages)

def upload_file(uploaded_files, catalog, schema, volume):
  return asyncio.run(upload_file_async(uploaded_files, catalog, schema, volume))

# COMMAND ----------

def question_answer(question, catalog, schema, volume, image):
  
  embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
  new_db = FAISS.load_local(f"/Volumes/{catalog}/{schema}/{volume}/llama", embedding_function, allow_dangerous_deserialization=True)
  retriever = new_db.as_retriever()
  dbrx_chat_model = ChatDatabricks(endpoint="databricks-meta-llama-3-1-405b-instruct", max_tokens = 4000)
  
  qa_chain = RetrievalQA.from_chain_type(llm=dbrx_chat_model,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)
  output = qa_chain(question)
  page_list = []
  
  for doc in output['source_documents']:
                      tmp=doc.metadata['page'] + 1
                      page_list.append(tmp)

  doc_pages = ','.join(set([str(pgno) for pgno in page_list]))
  doc_name = ','.join(set([str(doc.metadata['source']) for doc in output['source_documents']]))
  
  doc_content = '.\n'.join(set([str(doc.page_content) for doc in output['source_documents']]))
    
  return output['result'], doc_pages, doc_name, doc_content

# COMMAND ----------

# MAGIC %md
# MAGIC ####Create Gradio App

# COMMAND ----------

js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

with gr.Blocks(js=js_func) as demo:
  
  with gr.Tabs():
    with gr.TabItem("Upload Documents 📁"):
      catalog = gr.Textbox(label="Catalog", placeholder="Enter catalog name", container=True)
      schema = gr.Textbox(label="Schema", placeholder="Enter schema name", container=True)
      volume = gr.Textbox(label="Volume", placeholder="Enter Volume Name", container=True)
      
      source_files = gr.Files(label="Upload Source Files", file_types=["pdf"])
      output_text = gr.Text(label="Output", interactive = False)

      source_files.change(fn=upload_file, inputs=[source_files, catalog, schema, volume], outputs=output_text)

    with gr.TabItem("RAG Chatbot 💬"):
      question_input = gr.Textbox(label="Ask a Question", placeholder="Examples: What is the leave policy? How do I claim insurance? How do I ingest data at scale?")
      gr.Interface(
        fn=question_answer,
        inputs=[question_input, catalog, schema, volume, gr.Image(value='https://raw.githubusercontent.com/Vishesh8/databricks-tests/main/demo-images/rag_diag.png')],
        outputs=[
            gr.Textbox(label='Chatbot Response'),
            gr.Textbox(label='Document Page'),
            gr.Textbox(label='Document Location'),
            gr.Textbox(label='Document Context')
        ],
        title= "LLM RAG Chatbot - Powered By Llama 3.1 (405B)",
      )

# update debug to True to run in debug mode
demo.launch(share=True, debug=False)

# COMMAND ----------


