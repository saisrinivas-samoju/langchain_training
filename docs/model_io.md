# Model IO

## Implementation

### Steps:

#### API key loading
```py
import os
import warnings
warnings.filterwarnings('ignore')

with open('../openai_api_key.txt', 'r') as f:
    api_key = f.read()
    
os.environ['OPENAI_API_KEY'] = api_key

# os.getenv('OPENAI_API_KEY')
```
---

#### Load the text completion model
```py
from langchain.llms import OpenAI
llm = OpenAI()
```
---

#### Single Prompt
```py
prompt = "The impact of the globalization on diverse cultures can be explained as:"

response = llm(prompt=prompt)

response
```
```
> '\n\n1. Homogenization of Cultures: Globalization has led to the spread of Western culture and values across the world, ...
```

```py
print(response)
```
```
> 1. Homogenization of Cultures: Globalization has led to the spread of Western culture and values across the world, ...
```
---

#### Multiple prompts
```py
prompts = [
    "The impact of the globalization on diverse cultures can be explained as:",
    "Ecosystems maintains biodiversity as follows:"
]

response = llm.generate(prompts=prompts)

response
```
```
> LLMResult(generations=[[Generation(text='\n\n1. Cultural Homogenization: One of the major impacts of globalization on diverse cultures is the ...
```

```py
print(response.generations[0][0].text)
```
```
> 1. Cultural Homogenization: One of the major impacts of globalization on diverse ...
```

```py
# Print individual responses
for gen_list in response.generations:
    gen = gen_list[0]
    text = gen.text
    print(text)
    print("-"*50)
```
```
> 1. Cultural Homogenization: One of the major impacts of globalization on diverse ...
```

#### LLM usage Information
```py
response.llm_output
```
```
> {'token_usage': {'completion_tokens': 512,
'prompt_tokens': 21,
'total_tokens': 533},
'model_name': 'gpt-3.5-turbo-instruct'}
```
---

#### Response Caching
```py
from langchain.globals import set_llm_cache

# In memory caching

from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())

# SQLite caching

from langchain.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path='../models/cache.db'))
```
With this, your responses for the same prompts and parameters will be cached. That means, whenever you run the LLM with the previously ran prompts and parameters, your prompt won't hit the LLM, instead it will get the response from the cache memory.

Example: Let's get the response from the LLM for a random prompt
```py
response = llm("Give all the details about Bali...")
# time: 2.8s
```
When we run the same command again, after running the caching code
```py
response = llm("Give all the details about Bali...")
# time: 0.0s
```
---
#### Schema

    * SystemMessage: Role assigned to the AI.
    * HumanMessage: Human request or the prompt.
    * AIMessage: AI Response as per it's role to the Human request.

```py
from langchain.schema import SystemMessage, HumanMessage

response = chat(messages = [HumanMessage(content='What is the longest river in the world?')])

response # Response is an AIMessage
```
```
> AIMessage(content='The longest river in the world is the Nile River, which flows through northeastern Africa for about 4,135 miles (6,650 kilometers).')
```

```py
# Adding system message

messages = [
    SystemMessage(content='Act as a funny anthropologist'),
    HumanMessage(content="The impact of the globalization on diverse cultures can be explained as:")
]

response = chat(messages=messages)

response
```
```
> AIMessage(content="Ah, yes, the fascinating topic of globalization and its impact on diverse
```
---
#### Parameters

    [Click Here](https://platform.openai.com/docs/api-reference/chat/create) for the official documentation

```py
response = chat(
    messages=[
        SystemMessage(content='You are an angry doctor'),
        HumanMessage(content='Explain the digestion process in human bodies')
    ],
    model = "gpt-3.5-turbo", # Model for generation,
    temperature=2, # [0, 2] Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    presence_penalty=2, # [-2.0, 2.0]  increasing the model's likelihood to talk about new topics.
    max_tokens=100
)

print(response.content)
```

```
>   Ugh Cyril hung increased values Guards gala? Buck through ik St battleground
```
---
#### Few Shot Prompting
```py
from langchain.schema import AIMessage

system_message = "You are a funny doctor"

patient_dialogue1 = "Doctor, I have been feeling a bit under the weather lately."
sample_response1 = "Under the weather? Did you try checking the forecast before stepping out? You might need a weather app prescription!"

patient_dialogue2 = "My throat has been sore, and I have a cough."
sample_response2 = "The classic sore throat symphony! I recommend a strong dose of chicken soup and a dialy karaoke session. Sing it out, and your throat will thank you."

patient_dialogue3 = "I have a headache."
sample_response3 = "Headache, you say? Have you tried negotiating with it? Maybe it's just looking for a better job inside your brain!"

messages = [
    # SystemMessage(content=system_message),
    
    HumanMessage(content=patient_dialogue1),
    AIMessage(content=sample_response1),
    
    HumanMessage(content=patient_dialogue2),
    AIMessage(content=sample_response2),
    
    HumanMessage(content=patient_dialogue3),
    AIMessage(content=sample_response3),
    
    HumanMessage(content='I have a stomach pain')
]

response = chat(messages=messages)

print(response.content)
```

```
>   Stomach pain, huh? Maybe your stomach is just trying to tell a joke! Have you tried asking it to lighten up a bit?
```

---
## Exercise
* Create a cross-questioning bot with and without a system prompt
* Create a bad comedian bot that tries to crack jokes on every single thing that you say playing with the words in your dialogue.
---
## Tasks
* Write a blog on few shot prompting
* Create a GitHub account
---