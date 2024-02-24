# Model IO

## Implementation - Part 1

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

## Implementation - Part 2

### Steps:

#### Prompt Templating - Text Completion models

```py
# loading the models
import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import warnings
warnings.filterwarnings("ignore")

with open('../openai_api_key.txt', 'r') as f:
    os.environ['OPENAI_API_KEY'] = f.read()
    
llm = OpenAI()
chat = ChatOpenAI()

# Cache

from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())
```
##### Prompt templating - format strings

```py
prompt_template = "write an essay on {topic}"

prompt = prompt_template.format(topic='data science')

prompt
```
```
> 'write an essay on data science'
```

```py
print(llm(prompt_template.format(topic='science')))
```
```
> 

Science is a systematic and logical approach to understanding the natural world. It is a method of acquiring knowledge through observation, experimentation, and analysis. ...
```

##### Prompt templating - f-string literals
```py
topic = 'data science' # Need a global variable

prompt = f"Write an essay on {topic}"

prompt
```

```
> 'Write an essay on data science'
```
```py
# To use a local variable, create a function

def get_prompt(topic):
    
    prompt = f"Write an essay on {topic}"
    
    return prompt

get_prompt(topic='data science')
```
```
> 'Write an essay on data science'
```
These approaches won't scale up when we work with complex tasks like chains.

Let's learn how to use prompt templates in langchain

Prompt templating using langchain prompt template

##### Prompt templating - text completion models
```py
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=['topic'],
    template = "Write an essay on {topic}"
)

prompt = prompt_template.format(topic='data science')

prompt
```
```
> 'Write an essay on data science'
```
Another prompt with more inputs
```py
prompt_template = PromptTemplate(
    input_variables=['topic', 'num_words'],
    template = "Write an essay on {topic} in {num_words} words"
)

prompt = prompt_template.format(topic='data science', num_words=200)

prompt
```
```
> 'Write an essay on data science in 200 words'
```
For the same prompt_tempate, if you put a placeholder for the input_variable, it would still work the same way.
```py
prompt_template = PromptTemplate(
    input_variables=[],
    template = "Write an essay on {topic} in {num_words} words"
)

prompt = prompt_template.format(topic='data science', num_words=200)

prompt
```
```
> 'Write an essay on data science in 200 words'
```

```py
response = llm(prompt)

print(response)
```
```
> 

Data science is an interdisciplinary field that combines techniques and tools from statistics, mathematics, computer science, and information science to extract useful insights and knowledge from large and complex datasets. ...
```

##### Serialization
```py
prompt_template
```
```
> PromptTemplate(input_variables=['num_words', 'topic'], template='Write an essay on {topic} in {num_words} words')
```
Saving the prompt templates
```py
prompt_template.save("../output/prompt_template.json")
```
Loading the prompt templates
```py
from langchain.prompts import load_prompt

loaded_prompt_template = load_prompt('../output/prompt_template.json')

loaded_prompt_template
```
```
> PromptTemplate(input_variables=['num_words', 'topic'], template='Write an essay on {topic} in {num_words} words')
```
##### Prompt templating - chat completion models

Using format strings or f-string literals with langchain schema objects

format strings
```py
prompt_template = "Write a essay on {topic}"

system_message_prompt = SystemMessage(prompt_template.format(topic = "data science"))

system_message_prompt
```
f-string literals
```py
topic = "data science"

prompt_template = f"Write a essay on {topic}"

system_message_prompt = SystemMessage(prompt_template)

system_message_prompt
```
Issue: We are defining our inputs way ahead while using this type of prompt templating or making the inputs as global variables

Prompt templating using langchain prompt template

Starting with a simple Human Message Prompt Template

```py
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate

human_template = "Write an essay on {topic}"

human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])

prompt = chat_prompt.format_prompt(topic='data science')

prompt
```
```
> ChatPromptValue(messages=[HumanMessage(content='Write an essay on data science')])
```

To get the messages from teh ChatPromptValue

```py
# messages = prompt.to_messages()
messages = prompt.messages

messages
```
```
> [HumanMessage(content='Write an essay on data science')]
```
Getting the response from the chat model
```py
response = chat(messages=messages)

response
```
```
> AIMessage(content="Data science is a rapidly growing field that involves the collection, analysis, and interpretation...
```

Similarly, let's do it with Other schema of messages
```py
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate
```

System Message Prompt Template
```py

system_template = "You are a nutritionist"
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
system_message_prompt
```
```
> SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a nutritionist'))
```

Human Message Prompt Template
```py

human_template = "Tell the impact of {food_item} on human body when consumed regularly"
human_message_prompt = HumanMessagePromptTemplate.from_template(template=human_template)
human_message_prompt
```

```
> HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['food_item'], template='Tell the impact of {food_item} on human body when consumed regularly'))
```
Chat Prompt Template
```py
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chat_prompt
```
```
> ChatPromptTemplate(input_variables=['food_item'], messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a nutritionist')), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['food_item'], template='Tell the impact of {food_item} on human body when consumed regularly'))])
```

```py
prompt = chat_prompt.format_prompt(food_item='rice')

prompt
```
```
> ChatPromptValue(messages=[SystemMessage(content='You are a nutritionist'), HumanMessage(content='Tell the impact of rice on human body when consumed regularly')])
```

Chat Prompt Value to messages to pass to the chat model

```py
messages = prompt.to_messages()

messages
```
```
> [SystemMessage(content='You are a nutritionist'),
 HumanMessage(content='Tell the impact of rice on human body when consumed regularly')]
```
```py
response = chat(messages=messages)

response
```
```
> AIMessage(content="Rice is a staple food for many people around the world and can provide several health benefits when consumed regularly as part of a balanced diet. ...
```
