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

## Implementation - Part 2
### Text Splitter
#### Split by character

Reading the data

```python
filepath = "../datasets/Harry Potter 1 - Sorcerer's Stone.txt"

with open(filepath, 'r') as f:
    hp_book = f.read()
    
print("Number of characters letters in the document:", len(hp_book))
print("Number of words in the document:", len(hp_book.split()))
print("Number of lines in the document:", len(hp_book.split("\n")))
```
      Number of characters letters in the document: 439742
      Number of words in the document: 78451
      Number of lines in the document: 10703

To understand the how the number of characters if we use any separator manually

```python
from collections import Counter

line_len_list = []

for line in hp_book.split("\n"):
    curr_line_len = len(line)
    line_len_list.append(curr_line_len)
    
Counter(line_len_list) # It show how many those chunks with the same character length is present
```

      > Counter({37: 57,
            0: 3057,
            11: 38,
      ...
            4: 15,
            3: 9,
            2: 1})

#### Character Text Splitter

Splitting the text at a specific character only if the chunk exceeds the given chunk size

```python
from langchain.text_splitter import CharacterTextSplitter

def len_func(text): # In this case, you can just use >len<
    return len(text)

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1200,
    chunk_overlap=100,
    length_function=len_func,
    is_separator_regex=False
)

para_list = text_splitter.create_documents(texts=[hp_book])

para_list
```

      > [Document(page_content="Harry Potter and the Sorcerer's Stone\n\n\nCHAPTER ONE\n\nTHE BOY WHO LIVED
      ...
      I\'m going to have a lot of fun with Dudley this summer...."\n\nTHE END')]

To add metadata for the document objects

```python
first_chunk = para_list[0]

# Just assign/reassign
first_chunk.metadata = {"source": filepath}

first_chunk.metadata
```
      > {'source': "../datasets/Harry Potter 1 - Sorcerer's Stone.txt"}

What if the text exceeds the chunk length and there is not separator to chunk the text?

```python
# Adding the extra line
extra_line = " ".join(['word']*500)

para_list = text_splitter.create_documents(texts = [extra_line + hp_book])

# checking the length of the first line as the extra line is added there
first_chunk_text = para_list[0].page_content

len(first_chunk_text)
```
      Created a chunk of size 2536, which is longer than the specified 1200
      > 2536

Can we add multiple separators to make it working better?

That's where Recursive Character Text Splitter comes in.

#### Recursive Character Splitter

It tries to split on them in order until the chunks are small enough.
The default list is <code>["\n\n", "\n", " ", ""]</code>. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ' '],
    chunk_size = 200,
    chunk_overlap = 100,
    length_function = len_func,
    is_separator_regex=False
)

# Here, the split first happens at "\n\n", if the chunk size exceeds, it will move to the next separator, if it still exceeds, it will move to the next separator which is a " ".

chunk_list = text_splitter.create_documents(texts = [hp_book])

chunk_list
```
      > [Document(page_content='CHAPTER ONE\n\nTHE BOY WHO LIVED'),
      ...]

Let's see how this chunking process work in the previous scenario

```python
chunk_list = text_splitter.create_documents(texts = [extra_line + hp_book]) # Adding the extra line

chunk_list
```

      > [Document(page_content='word word word word word word ...

      ...]

The text got chunked at spaces to maintain the chunk size in the first line.

#### Split by tokens

tiktoken is a python library developed by openAI to count the number of tokens in a string without making an API call.
```console
pip install tiktoken
```

Splitting based on the token limit

```python
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n\n", 
    chunk_size=1200, 
    chunk_overlap=100, 
    is_separator_regex=False,
    model_name='text-embedding-3-small',
    encoding_name='text-embedding-3-small', # same as model name
)

doc_list = text_splitter.create_documents([hp_book])

doc_list
```
      > [Document(page_content="Harry Potter and the Sorcerer's Stone\n\n\nCHAPTER ONE\n\nTHE BOY WHO LIVED
      ...
      I\'m going to have a lot of fun with Dudley this summer...."\n\nTHE END')]

The model name here refers to the model used for calculating the tokens.

To split the text and return the text chunks

```python
line_list = text_splitter.split_text(hp_book)

line_list
```
      > ['Harry Potter and the Sorcerer\'s Stone\n\n\nCHAPTER ONE\...
      ...Dudley this summer...."\n\nTHE END']

If you want to convert the split text into list of document objects
```python
from langchain.docstore.document import Document

doc_list = []

for line in line_list:
    curr_doc = Document(page_content=line, metadata={"source": filepath})
    doc_list.append(curr_doc)
    
doc_list
```
      > [Document(page_content="Harry Potter and the Sorcerer's Stone\n\n\nCHAPTER ONE\n\nTHE BOY WHO LIVED
      ...
      I\'m going to have a lot of fun with Dudley this summer...."\n\nTHE END')]

# Code Splitting

Let's learn a generic way of splitting code that's written in any language. For this let's convert the previous peer_review function code into text.

```python
python_code = """def peer_review(article_id):
    chat = ChatOpenAI()
    loader = ArxivLoader(query=article_id, load_max_docs=2)
    data = loader.load()
    first_record = data[0]
    page_content = first_record.page_content
    title = first_record.metadata['Title']
    summary = first_record.metadata['Summary']
    
    summary_list = []
    for record in data:
        summary_list.append(record.metadata['Summary'])
    full_summary = "\n\n".join(summary_list)
    
    system_template = "You are a Peer Reviewer"
    human_template = "Read the paper with the title: '{title}'\n\nAnd Content: {content} and critically list down all the issues in the paper"

    systemp_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([systemp_message_prompt, human_message_prompt])
    prompt = chat_prompt.format_prompt(title=title, content=page_content)

    response = chat(messages = prompt.to_messages())

    return response.content"""
```

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=50,
    chunk_overlap=10
)

text_splitter.create_documents(texts = [python_code])
```

      > [Document(page_content='def peer_review(article_id):'),
      Document(page_content='chat = ChatOpenAI()'),
      ...
      Document(page_content='= prompt.to_messages())'),
      Document(page_content='return response.content')]

Similar to python code, you can also split any the code in programming language. For example: To split javascript code use <code>Language.JS</code>

#### Embeddings

