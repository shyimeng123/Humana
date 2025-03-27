# Humana

This repo is a simple demo of a chatbot based on RAG that retrieves a single PDF research paper.
It uses Open AI gpt 4-o to build the embeddings and process the answer.
Here is an example of the terminal running of the python script:
You could interact with the std in/ std out
![Screenshot](https://github.com/shyimeng123/Humana/blob/3fcdf7b2d2d497f19bb232252221d123e496634e/TerminalRun.png)

Since I paid for the tokens through my personal account, I did NOT try to run and evaluate massively. Below are some further points:

# UI
For UI, we can use streamlit or write an app to modulize. Here is an example of streamlit UI that I wrote before to compare different embeddings for a given input text. ![screenshot](https://github.com/shyimeng123/Humana/blob/4a99fb155d584c36175e6cdd09cc1de6eedcc4cd/SampleApp.png)
(https://github.com/shyimeng123/EmbeddingComparisonApp/blob/aeb661c691bc63d15bc4b165d835e4d3f02210e5/README.md) After knowing the product/business needed, I can create the library and UI for general usage.

# Optimization
One can build a chat bot within 2 days : ) The key challenge is to optimize through the following main aspects of quality, cost and latency.

## 1. Quality
### Input
For input, when doining this project, I tried to use the default pdf loader from langchain initially but found for some of the pdf content it cannot seperate the words correctly. Optical Character Recognition (OCR) is an approach to convert the pdf with higher resolution. Also there are lots of different approached such as unstructured pdf loader provided by langchain and many others. 

### Prompting
We can use zero-shot, few-shot, chain of thought, graph of thought etc depending on the usage and buisness goal and even to build an agent

### Output
The output examples are given in the ChatBot2.py, at the end of there are 3 instances. There are also a lot to optimize such as stem the key concepts and may further ranking to retrive. Right now the quality is limited by the pdf loader/aplitter---it contains strings that are not well splited. There should be some post processing process to better present the answers.

### Embedding
Embedding is crucial to improve the retrieval quality. There are a lot of models such as open AI, meta Llamma, Gemini etc. Besides those llm, some open source embedding models such as microsoft E5 embedding model and the one you shared for medical knowledge graph based embedding can be used here.

For the knowledge graph implementation, we may need to build a classifier such as rule based/logistic regression to decide if we would like to "fetch" from the knowledge base or not. If it is some general question, then we wont fetch the biological/medical terminology. And then we will need to build the vector store for future usage.

Moreover, it can be finetuned from the model base to improve the performance. I myself had experience finituning open sources E5 model with domain specific data. That one uses NCE info loss so that we only need to have positive, negative query/answer pairs. 
  
## 2.Cost and Latency

Chat bots are expensive and the cost is depending on the api such as open AI, Llama, Deekseek etc. Firstly we may do an ablation analyis to compare/evaluate the cost for different LLM platform.

To optimize the cost, we need to carefuly decide which tokens to input and which tokens to output. Some rule based/MLE model can be applied on top of it.
Then it may encounter another issue--latency. If there are too many adds on, the response time may be slow, There are many things to work on for this aspects.

# Evaluation
Evaluation is another important aspect. And the design of it relies on the business situation.

## Offline evaluation:
### Benchmark Data
There are some general NLP evaluation data set such as BEIR and MTEB for information retrieval tasks. Besides, we can build, like created some human annotaed benchmark data, lets say maybe with thousands examples.

### Metrics
We can also collect past successul chat interactive data and come up with metrics based on user feedback like satisfied/unsatisfied

### LLM
We can always write prompts to ask GPT to provide us a relevance score, lets say from 1,2,3,4,5 for a chat session

## Online evaluation
### Metrics Related to User Engagement:
carefully design some metrics like interaction time, stop time, if repeatdly asking some question etc.

### Success or not success
depend on the business target, we may have indicator that does the chatbot/agent successfuly solve the problem




