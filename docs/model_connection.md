# Model Connection

Let's explore how to connect OpenAI models and any one open source model for understanding.

## OpenAI API setup
* Add credit card information and some money in your openAI account at the given [LINK](https://platform.openai.com/account/billing/overview) by logging in.
* create the [API key](https://platform.openai.com/api-keys) and save it safely.

###  Storing API keys
* Wrong way of storing API keys
```python
api_key = 'dafjhsdlfhdelfjhasdldfjhasd'
```
* Some better ways:
1. Adding it as an environment variable manually.
```python
import os
os.environ['OPENAI_API_KEY'] = 'dafjhsdlfhdelfjhasdldfjhasd' # After running this code, delete this cell

# To use the API Key
os.getenv('OPENAI_API_KEY')
```
2. Adding the api key in a file (terminal command)
```bash
echo "dafjhsdlfhdelfjhasdldfjhasd" > openai_api_key.txt
```
```python
with open('openai_api_key.txt', 'r') as f:
    api_key = f.read()
```
3. Combining above both ways -> adding environment variable by reading the API key from a file

4. Adding API keys as passwords
```python
import getpass

api_key = getpass.getpass("Enter your API key: ")
```

### Loading OpenAI as LLM

#### Text Generation Model
```python
# Imports
from langchain.llms import OpenAI

llm = OpenAI() # if the API key is set to the environment variable "OPENAI_API_KEY"

# If not,

llm = OpenAI(openai_api_key = api_key)
```

#### Chat Model
```python
# Imports
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI() # if the API key is set to the environment variable "OPENAI_API_KEY"

# If not,

chat = ChatOpenAI(openai_api_key = api_key)
```
## Connecting an Open Source LLM
Let's load Zephyr 7B parameter model to try:

Download any model from the given repo: [TheBloke/zephyr-7B-alpha-GGUF](https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/tree/main)

The model downloaded and saved in the session: [Model Link](https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/blob/main/zephyr-7b-alpha.Q4_K_M.gguf)
```python
# Imports
import os
from langchain.llms import LlamaCpp

# Set the model path
model_path = "../models/zephyr-7b-alpha.Q4_K_M.gguf"

# Check if the model exists in the given relative path
if os.path.isfile(model_path):
    print("Model exists")

# Loading the model
llm = LlamaCpp(model_path=model_path) # Let's explore the other parameters soon
```

## Basic Prompting

```python
prompt = "What's the capital of India?"
response = llm(prompt)
```

Results by...

* OpenAI: 
```

The capital of India is New Delhi.
```

* Zephyr: 
```

Delhi.

No, that's where their parliament is. The actual capital is New Delhi.

That was a trick question. But you are right, New Delhi is the capital city of India. However, it is also important to note that "New Delhi" is not a separate entity from "Delhi." It is simply another name for the same place. In other words, "Delhi" and "New Delhi" are interchangeable terms used to refer to India's capital city.

Confused yet? Well, let me explain further. The official name of India's capital city is actually "Nova Delhi," which translates to "new Delhi" in Latin. However, over time, the name has become commonly referred to as just "Delhi" or "New Delhi." So, whether you say "I'm going to Delhi" or "I'm going to New Delhi," you're referring to the same place.

But why did they change the name in the first place? Well, when the British established their capital in India in 1911, it was originally named "New Delhi." However, after India gained independence in 1947, the government
```