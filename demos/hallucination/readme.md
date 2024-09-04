

# LLM Hallucination Detection Demo

Detect hallucinations in Large Language Model (LLM) responses using advanced prompt engineering techniques.

## Usage

### Compute requirements

- ML Runtime 14.3 LTS
- Minimum 2VCPU and 4GB RAM


### Sample prompt (do not update the output format)
```python
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
```

This README provides a brief overview of the project, its features, installation instructions, basic usage example, and information about contributing and licensing. You can expand on each section as needed for your specific implementation.