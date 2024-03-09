# Retrieval

Large language models are trained on massive datasets, but they don't know everything. That's where Retrieval Augmented Generation (RAG) comes in, and LangChain has you covered with all the basics to Advanced bulding blocks:

<img src='https://python.langchain.com/assets/images/data_connection-95ff2033a8faa5f3ba41376c0f6dd32a.jpg' width=750>

1. **Document loaders:**
   Learn how to put your own information into the training mix.

2. **Text Splitters:**
   Find out how to tweak your data to make the model understand better.

3. **Text Embedding Models:**
   Discover ways to seamlessly include your information within the model.

4. **Vector Stores:**
   Explore ways to save these tweaks for quick and efficient retrieval.

5. **Retrievers:**
   Learn how to ask the model for the information you stored.

LangChain breaks down each step, making it easy for you to make the most of RAG with your language models.

---

## Implementation - Part 1

### Document Loaders
#### CSV Loader

```python
from langchain.document_loaders import CSVLoader
loader = CSVLoader(file_path='../datasets/sns_datasets/titanic.csv') # Lazy Loader
loader
```
      > <langchain_community.document_loaders.csv_loader.CSVLoader at 0x147b62e5760>
   
```python
data = loader.load()

data
```
      > [Document(page_content='survived: 0\npclass: 3\nsex: male\nage: 22.0\nsibsp: 1\nparch: 0\nfare: 7.25\nembarked: S\nclass: Third\nwho: man\nadult_male: True\ndeck: \nembark_town: Southampton\nalive: no\nalone: False', metadata={'source': '../datasets/sns_datasets/titanic.csv', 'row': 0}), Document(page_content='survived: 1\npclass: 1\nsex: female\nage: 38.0\nsibsp: 1\nparch: 0\nfare: 71.2833\nembarked: C\nclass: First\nwho: woman\nadult_male: False\ndeck: C\nembark_town: Cherbourg\nalive: yes\nalone: False', metadata={'source': '../datasets/sns_datasets/titanic.csv', 'row': 1}), ...]
Python list of document objects; Each row in a separate document object

```python
data[0] # Single Row
```
      > Document(page_content='survived: 0\npclass: 3\nsex: male\nage: 22.0\nsibsp: 1\nparch: 0\nfare: 7.25\nembarked: S\nclass: Third\nwho: man\nadult_male: True\ndeck: \nembark_town: Southampton\nalive: no\nalone: False', metadata={'source': '../datasets/sns_datasets/titanic.csv', 'row': 0})

```python
type(data[0])
```
      > langchain_core.documents.base.Document

To get the document content
```python
print(data[0].page_content)
```
      >  survived: 0
         pclass: 3
         sex: male
         age: 22.0
         sibsp: 1
         parch: 0
         fare: 7.25
         embarked: S
         class: Third
         who: man
         adult_male: True
         deck: 
         embark_town: Southampton
         alive: no
         alone: False

To get the metadata

```python
print(data[0].metadata)
```
      > {'source': '../datasets/sns_datasets/titanic.csv', 'row': 0}

Specify a column name to identify the dataset

```python
data = CSVLoader(file_path='../datasets/sns_datasets/titanic.csv', source_column= 'sex').load()
```

#### HTML Loader

Similar syntax
```python
from langchain.document_loaders import UnstructuredHTMLLoader
loader = UnstructuredHTMLLoader('../datasets/harry_potter_html/001.htm')
data = loader.load()
data
```
      > [Document(page_content='A Day of Very Low Probability\n\nBeneath the moonlight glints a tiny fragment of silver, a fraction of a line…\n\n ...

Loading HTML documents with BeautifulSoup

```python
from langchain.document_loaders import BSHTMLLoader
loader = BSHTMLLoader('../datasets/harry_potter_html/001.htm')
data = loader.load()
data
```
      > A Day of Very Low Probability

      Beneath the moonlight glints a tiny fragment of silver, a fraction of a line…
      (black robes, falling)
      …blood spills out in litres, and someone screams a word. ...

This response is close to the content in the HTML file.

#### Markdown Loader

```python
from langchain.document_loaders import UnstructuredMarkdownLoader

md_filepath = "../datasets/harry_potter_md/001.md"

loader = UnstructuredMarkdownLoader(file_path=md_filepath)

data = loader.load()

data
```
      > [Document(page_content='A Day of Very Low Probability\n\nBeneath the moonlight glints a tiny fragment of silver ...

#### PDF Loader
```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader('../datasets/harry_potter_pdf/hpmor-trade-classic.pdf')

data = loader.load()

data
```
      [Document(page_content='Harry Potter and the Methods of Rationality', metadata={'source': '../datasets/harry_potter_pdf/hpmor-trade-classic.pdf', 'page': 0}),
      Document(page_content='', metadata={'source': '../datasets/harry_potter_pdf/hpmor-trade-classic.pdf', 'page': 1}), ...

#### Wikipedia

```python
from langchain.document_loaders import WikipediaLoader
loader = WikipediaLoader(query='India', load_max_docs=1)
data = loader.load()

data
```
      > [Document(page_content="India, officially the Republic of India (ISO: Bhārat Gaṇarājya), ...

Since we are only loading one document, the number of document objects in the response list is also 1
```python
len(data)
```
      > 1

To get the metadata
```python
# select the document object
data[0].metadata
```
      > {'title': 'India',
      'summary': "India, officially the Republic of India (ISO: Bhārat Gaṇarājya), ...",
      'source': 'https://en.wikipedia.org/wiki/India'}

Call the keys of the dictionary to get the specific information from the metadata.

Information from the metadata can be used to filter you data in the later stages.

#### ArXiv Loader

Loading the content from the famous scientific article publisher

To get the article IDs of any ArXiv papers, check the URL of the page or header of the page.

<img src="img/article_id_in_arXiv.png">

```python
from langchain_community.document_loaders import ArxivLoader

loader = ArxivLoader(query='2201.03916', load_max_docs=1) # AutoRL paper (article ID -> 2201.03916)

data = loader.load()

data
```
      > [Document(page_content='Journal of Artiﬁcial Intelligence Research 74 (2022) ...

```python
len(data) # since load_max_docs=1
```
      > 1

```python
print(data[0].page_content)
```  
      > Journal of Artiﬁcial Intelligence Research 74 (2022) 517-568
      Submitted 01/2022; published 06/2022
      Automated Reinforcement Learning (AutoRL):

Getting the metadata similar to the previous steps

```python
data[0].metadata
```
      > {'Published': '2022-06-02',
      'Title': 'Automated Reinforcement Learning (AutoRL): A Survey and Open Problems',
      'Authors': 'Jack Parker-Holder, Raghu Rajan, Xingyou Song, André Biedenkapp, Yingjie Miao, Theresa Eimer, Baohe Zhang, Vu Nguyen, Roberto Calandra, Aleksandra Faust, Frank Hutter, Marius Lindauer',
      'Summary': 'The combination of Reinforcement Learning (RL) ..."}

Let's connect the retrieved information to the LLM

```python
import os
from langchain_openai import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

with open("../openai_api_key.txt", 'r') as f:
    os.environ['OPENAI_API_KEY'] = f.read()

chat = ChatOpenAI()
set_llm_cache(InMemoryCache())
```

```python
# Setting up the prompt templates

from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

system_template = "You are Peer Reviewer"
human_template = "Read the paper with the title: '{title}'\n\nAnd Content: {content} and critically list down all the issues in the paper"

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(messages=[system_message_prompt, human_message_prompt])
prompt = chat_prompt.format_prompt(title=data[0].metadata['Title'], content=data[0].page_content)
messages = prompt.to_messages()

response = chat(messages=messages)

print(response.content)
```
      > Overall, the paper "Attention Is All You Need" presents a novel model architecture, the Transformer, which is based solely on attention mechanisms and dispenses with recurrence and convolutions. The paper provides a detailed description of the model architecture, background information, model variations, training process, and results in machine translation and English constituency parsing tasks. The paper also includes attention visualizations to illustrate the behavior of the attention heads.

      Here are some key points to consider for a critical review of the paper: ...

```python
def peer_review(article_id):
    chat = ChatOpenAI(max_tokens=500)
    loader = ArxivLoader(query=article_id, load_max_docs=1)
    data = loader.load()
    page_content = data[0].page_content
    title = data[0].metadata['Title']
    summary = data[0].metadata['Summary']

    system_template = "You are Peer Reviewer"
    human_template = "Read the paper with the title: '{title}'\n\nAnd Content: {content} and critically list down all the issues in the paper"

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(messages=[system_message_prompt, human_message_prompt])

    try:
        prompt = chat_prompt.format_prompt(title=title, content=page_content) # Suggest not to go with this
        messages = prompt.to_messages()
        response = chat(messages=messages)
    except:
        prompt = chat_prompt.format_prompt(title=title, content=summary)
        messages = prompt.to_messages()
        response = chat(messages=messages)

    return response.content
```


```python
print(peer_review(article_id='2201.03514')) # Black-Box Tuning for Language-Model-as-a-Service
```

      > After reviewing the paper titled 'Black-Box Tuning for Language-Model-as-a-Service', I have identified several issues in the paper that need to be addressed before it can be considered for publication:

      1. Lack of Clarity in Problem Statement: The paper does not clearly define the problem statement or research question it aims to address. It is unclear why optimizing task prompts through black-box tuning for Language-Model-as-a-Service (LMaaS) is important or how it contributes to the existing body of knowledge.


