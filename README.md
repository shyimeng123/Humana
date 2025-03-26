# Humana

This repo is a simple demo of a chatbot based on RAG that retrieves a single PDF research paper.
It uses Open AI gpt 4-o to build the embeddings and process the answer.
Here is an example of the terminal running of the python script:


Since I paid for the tokens through my personal account, I did NOT try to run and evaluate massively. Below are some further points:

# UI
For UI, we can use streamlit or write an app to modulize. Here is an example of streamlit UI that I wrote before to compare different embeddings for a given input text. After knowing the product/business needed, I can create the library and UI for general usage.

# Optimization
One can build chat bot within 2 days : ) The key challenge is to optimize through the following main aspects:

## 1. Quality
### Input
For input, when doining this project, I tried to use the default pdf loader from langchain initially but found for some of the pdf content it cannot seperate the words correctly. Optical Character Recognition (OCR) is an approach to convert the pdf with higher resolution. Also there are lots of different approached such as unstructured pdf loader provided by langchain and many others. 

### Prompting
We can use zero-shot, few-shot, chain of thought, graph of thought etc depending on the usage and buisness goal and even to build an agent

### Output
The output examples are given in the ChatBot2.py, at the end of there are 3 instances. There are also a lot to optimize such as stem the key concepts and may further ranking to retrive

### Embedding
  
## 2.Cost
