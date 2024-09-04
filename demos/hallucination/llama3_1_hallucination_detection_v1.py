# Databricks notebook source
# DBTITLE 1,Install required libraries
# MAGIC
# MAGIC %pip install -U --quiet databricks-sdk langchain-core langchain-community==0.2.6 gradio pymupdf 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Prompt for Hallucination detection
PROMPT = """
Given the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer and determine whether it is faithful to the contents of the DOCUMENT. The ANSWER must not offer new information beyond the context provided in the DOCUMENT. The ANSWER also must not contradict information provided in the DOCUMENT. Output your final verdict by strictly following this format: "PASS" if the answer is faithful to the DOCUMENT and "FAIL" if the answer is not faithful to the DOCUMENT. List only the top relevant keywords only from the ANSWER (not from DOCUMENT) that lead to your conclusion.

--
QUESTION (THIS DOES NOT COUNT AS BACKGROUND INFORMATION):
{question}

--
DOCUMENT:
{context}

--
ANSWER:
{answer}

--

Your output should be a single record only and in a valid JSON format with the keys "REASONING", "SCORE" and "KEYWORDS" and respective values within double quotes. The reasoning needs to be a single paragraph not a list. You MUST follow the below format for your output:
{{"REASONING": "<<your reasoning as bullet points>>", "SCORE": "<<your final score>>", "KEYWORDS": "<<top keywords from answer that led to your conclusion>>"}}
"""

# COMMAND ----------

# DBTITLE 1,helper functions to read PDF file
import pymupdf
from pathlib import Path

def get_filetype(filename):
    return filename.split(".")[-1]

def extract_text_pymupdf(file):
    with pymupdf.open(file) as pdf_or_txt:
        text = ""
        for page in pdf_or_txt:
            text += page.get_text()
    return text

# COMMAND ----------

# DBTITLE 1,helper function to chat LLM
from langchain_community.chat_models import ChatDatabricks

def invoke_llm(prompt):
  llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-405b-instruct",
         max_tokens = 500)
  ai_message = llm.invoke(prompt)
  return ai_message.content

# COMMAND ----------

eg_context = """ Databricks is a cloud-based platform that provides a fast, easy, and collaborative way to process large-scale data using Apache Spark. It was founded by the creators of Apache Spark and provides a managed environment for data engineering, data science, and data analytics.\n\nDatabricks allows users to:\n\n1. **Process large-scale data**: Databricks provides a scalable infrastructure to process massive amounts of data using Apache Spark, which is a unified analytics engine for large-scale data processing.\n2. **Write code in multiple languages**: Users can write code in languages like Python, R, SQL, and Scala to analyze and process data.\n3."""

eg_question="what is databricks majorly known for?"

eg_fail_answer = "Databricks is known for the ease of implementing ML and AI"

eg_pass_answer = "Databricks is a cloud-based platform that provides a fast, easy, and collaborative way to process large-scale data using Apache Spark. It was founded by the creators of Apache Spark and provides a managed environment for data engineering,"

# COMMAND ----------

fail_prompt = PROMPT.format(question=eg_question, context=eg_context, answer=eg_fail_answer)

pass_prompt = PROMPT.format(question=eg_question, context=eg_context, answer=eg_pass_answer)

# COMMAND ----------

invoke_llm(fail_prompt)

# COMMAND ----------

invoke_llm(pass_prompt)

# COMMAND ----------

import json
def llm_chat(filepath, input_question, output_response):
    extracted_file_text = ""
    if filepath is not None:
        name = Path(filepath).name
        filetype = get_filetype(name)
        # conditionals for filetype and function call
        if filetype == "pdf":
            extracted_file_text = extract_text_pymupdf(filepath)

        final_prompt = PROMPT.format(
          question=input_question, 
          context=extracted_file_text, 
          answer=output_response)

    response = invoke_llm(final_prompt)
    score = json.loads(response)['SCORE']
    reasoning = json.loads(response)['REASONING']
    keywords = json.loads(response)['KEYWORDS']
    return(score, keywords, reasoning)

# COMMAND ----------

import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown(
      """
      # Hallucination Detection Demo
      Upload a pdf document and ask any questions based on the document. 
      Use the buttons to answer the question and check if the answer is hallucinated or not.
      """)
    with gr.Row():
        with gr.Column():
            pdf_file = gr.File(label="Upload PDF File", type="filepath")
            input_qn = gr.Textbox(label="Input Question")
            llm_res = gr.Textbox(label="LLM Response")
            with gr.Row():
                  h_button = gr.Button("Check Hallucination")
        with gr.Column():
            pass_fail = gr.Textbox(label="Pass/Fail")
            keywords=  gr.Textbox(label="Keywords")
            reasoning = gr.Textbox(label="Reasoning")
    

    h_button.click(
      fn= llm_chat,      
      inputs=[pdf_file, input_qn, llm_res],  
      outputs=[pass_fail, keywords, reasoning]
    )
    

demo.launch(share=True, debug=True)



# COMMAND ----------


