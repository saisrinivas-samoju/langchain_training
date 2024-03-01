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
---
## Implementation - Part 3
### Steps
#### Output Parsers
Loading the language model and setting the cache
```py
import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
import warnings
warnings.filterwarnings('ignore')

with open('../openai_api_key.txt', 'r') as f:
    api_key = f.read()
    
os.environ['OPENAI_API_KEY'] = api_key

llm = OpenAI()
chat = ChatOpenAI()


set_llm_cache(InMemoryCache())
```

##### Steps to use the output parser
* format_instructions
* parse

Step 1: Create and instance of the parser
```py
from langchain.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()

output_parser
```
```
> CommaSeparatedListOutputParser()
```
Step 2: Get the format instructions
```py
output_parser.get_format_instructions()
```
```
> 'Your response should be a list of comma separated values, eg: `foo, bar, baz`'
```

Step 3: Send the instructions to the model
```py
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

human_template = "{user_request}\n{format_instructions}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
prompt = chat_prompt.format_prompt(user_request="What are the 7 wonders?", format_instructions=output_parser.get_format_instructions())

prompt
```
```
> ChatPromptValue(messages=[HumanMessage(content='What are the 7 wonders?\nYour response should be a list of comma separated values, eg: `foo, bar, baz`')])
```
```py
messages = prompt.to_messages()

response = chat(messages=messages)

print(response.content)
```
```
> Great Pyramid of Giza, Hanging Gardens of Babylon, Statue of Zeus at Olympia, Temple of Artemis at Ephesus, Mausoleum at Halicarnassus, Colossus of Rhodes, Lighthouse of Alexandria
```
Step 4: use the parser to parse the output
```py
output_parser.parse(response.content)
```
```
> ['Great Pyramid of Giza',
 'Hanging Gardens of Babylon',
 'Statue of Zeus at Olympia',
 'Temple of Artemis at Ephesus',
 'Mausoleum at Halicarnassus',
 'Colossus of Rhodes',
 'Lighthouse of Alexandria']
```
#### When parser fails?
```py
from langchain.output_parsers import DatetimeOutputParser

output_parser = DatetimeOutputParser()

format_instructions = output_parser.get_format_instructions()

print(format_instructions)
```
```
> Write a datetime string that matches the following pattern: '%Y-%m-%dT%H:%M:%S.%fZ'.

Examples: 0278-08-03T19:42:55.481110Z, 1567-04-05T01:30:42.197571Z, 0101-06-24T18:20:21.443663Z

Return ONLY this string, no other words!
```
```py
human_template = "{human_messsage}\n{format_instructions}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
prompt = chat_prompt.format_prompt(human_messsage="When was Jesus Christ born?", format_instructions=format_instructions)
messages = prompt.to_messages()

response = chat(messages=messages)

output = output_parser.parse(response.content)

output
```
```
> ---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
File d:\CodeWork\GitHub\langchain_training\.venv\lib\site-packages\langchain\output_parsers\datetime.py:50, in DatetimeOutputParser.parse(self, response)
     49 try:
---> 50     return datetime.strptime(response.strip(), self.format)
     51 except ValueError as e: ...
```
##### OutputFixingParser

```py
from langchain.output_parsers import OutputFixingParser

fixing_parser = OutputFixingParser.from_llm(parser=output_parser, llm=chat)

fixed_output = fixing_parser.parse(response.content)

fixed_output
```
```
> datetime.datetime(1, 1, 1, 0, 0)
```
Fixing might not always work, So let's try multiple times
```py
for chance in range(1, 10):
    try:
        fixed_output = fixing_parser.parse(response.content)
    except:
        continue
    else:
        break
    
fixed_output
```
```
> datetime.datetime(1, 1, 1, 0, 0)
```

#### Custom Parsers
##### Structured Output Parser
Define the response schema
```py
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

response_schemas = [
    ResponseSchema(name="answer", description="answer to the user's question"),
    ResponseSchema(
        name="source",
        description="source used to answer the user's question, should be a website.",
    ),
]
```

Define the output parser

```py
from langchain.output_parsers import StructuredOutputParser

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
output_parser
```
```
> StructuredOutputParser(response_schemas=[ResponseSchema(name='answer', description="answer to the user's question", type='string'), ResponseSchema(name='source', description="source used to answer the user's question, should be a website.", type='string')])
```

Get the format instructions

```py
format_instructions = output_parser.get_format_instructions()
format_instructions
```
```
> 'The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":\n\n```json\n{\n\t"answer": string  // answer to the user\'s question\n\t"source": string  // source used to answer the user\'s question, should be a website.\n}\n```
```

Get the response

```py
human_template = "{human_message}\n{format_instructions}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
prompt = chat_prompt.format_prompt(human_message = "What's the world's largest man made structure?", format_instructions=format_instructions)
messages = prompt.to_messages()

response = chat(messages=messages)

output = output_parser.parse(response.content)

output
```
```
> {'answer': 'The Great Wall of China',
 'source': 'https://www.history.com/topics/great-wall-of-china'}
```

Let's look at the more powerful way of creating custom parser

#### PydanticOutputParser

Let's quickly learn about pydantic

Conventional pythonic way of building classes

```py
class Student:
    def __init__(self, name: str):
        self.name = name
        
john = Student(name='John')
john.name
```
```
> 'John'
```

Similarily
```py
jane = Student(name=1) # Taking int even after defining the name to be str
jane.name
```
```
> 1
```

```py
type(jane.name) # Returning int too

# Conventional approach doesn't have strict type validation
```
```
> int
```

Pydantic has simple syntax with strict type validation

```py
from pydantic import BaseModel

class Student(BaseModel):
    name: str
    
jane = Student(name=1) # THIS WILL THROW AN ERROR

jane = Student(name='jane')
jane.name
```
```
> 'jane'
```

Let's get back to langchain

When we want our output to be in a specific class object format

First let's define the class

```py
from pydantic import BaseModel, Field
from typing import List

class Car(BaseModel):
    name: str = Field(description="Name of the car")
    model_number: str = Field(description="Model number of the car")
    features: List[str] = Field(description="List of features of the car")
```

create an instance of our custom parser

```py
from langchain.output_parsers import PydanticOutputParser

output_parser = PydanticOutputParser(pydantic_object=Car)

print(output_parser.get_format_instructions())
```
```
> The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema ...
```

Getting the response

```py
human_template = "{human_message}\n{format_instructions}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
prompt = chat_prompt.format_prompt(human_message='Tell me about the most expensive car in the world',
                                   format_instructions=output_parser.get_format_instructions())

response = chat(messages=prompt.to_messages())
output = output_parser.parse(response.content)

output
```
```
> Car(name='Bugatti La Voiture Noire', model_number='Divo', features=['1500 horsepower engine', '8.0-liter quad-turbocharged W16 engine', 'carbon fiber body', 'top speed of 261 mph'])
```
```py
type(output)
```
```
> __main__.Car
```

#### PydanticStructuredOutputParser
The new ChatOpenAI model from langchain_openai supports <code>with_structured_output</code> method, which can take the pydantic models built with **pydantic_v1** from **langchain_core**
```console
pip install langchain_openai
```
We will still be using <code>from langchain.chat_models import ChatOpenAI</code> for loading the chat model, to follow the standard structure of langchain (though it is depreciated)
```py
import os
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

with open('../openai_api_key.txt') as f:
    os.environ['OPENAI_API_KEY'] = f.read()

class Car(BaseModel):
    name: str = Field(description="Name of the car")
    model_number: str = Field(description="Model number of the car")
    features: List[str] = Field(description="List of features of the car")
    
model = ChatOpenAI()
model_with_structure = model.with_structured_output(Car)
model_with_structure.invoke('Tell me about the most expensive car in the world')
```

```
> Car(name='Bugatti La Voiture Noire', model_number='1', features=['Luxurious design', 'Powerful engine', 'Top speed of 261 mph', 'Exclusive and limited edition'])
```

#### Project ideas
* Real time text translation
* Text Summarization tool
* Q&A System
* Travel Planner
* Tweet Responder

#### Exercise
**Create a Smart Chef bot that can give you recipes based on the available food items you have in your kitchen.**

Let's build a gradio app
```py
import os
from typing import List
import gradio as gr
from pydantic import Field, BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

# Creating the instance of the chat model

with open('openai_api_key.txt', 'r') as f:
    api_key = f.read()
    
os.environ['OPENAI_API_KEY'] = api_key

chat = ChatOpenAI()

# Define the Pydantic Model

class SmartChef(BaseModel):
    name: str = Field(description="Name fo the dish")
    ingredients: dict = Field(description="Python dictionary of ingredients and their corresponding quantities as keys and values of the python dictionary respectively")
    instructions: List[str] = Field(description="Python list of instructions to prepare the dish")
    
# Get format instructions

from langchain.output_parsers import PydanticOutputParser

output_parser = PydanticOutputParser(pydantic_object=SmartChef)
format_instructions = output_parser.get_format_instructions()
format_instructions

def smart_chef(food_items: str) -> list:

    # Getting the response
    human_template = """I have the following list of the food items:

    {food_items}

    Suggest me a recipe only using these food items

    {format_instructions}"""

    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
    prompt = chat_prompt.format_prompt(
        food_items=food_items, format_instructions=format_instructions)

    messages = prompt.to_messages()
    response = chat(messages=messages)
    output = output_parser.parse(response.content)

    dish_name, ingredients, instructions = output.name, output.ingredients, output.instructions
    return dish_name, ingredients, instructions

# Building interface
with gr.Blocks() as demo:
    gr.HTML("<h1 align='center'>Smart Chef</h1>")
    gr.HTML("<h3 align='center'><i>Cook with whatever you have</i></h3>")
    inputs = [gr.Textbox(label='Enter the list of ingredients you have, in a comma separated text', lines=3, placeholder='Example: Chicken, Onion, Tomatoes, ... etc.')]
    generate_btn = gr.Button(value="Generate")
    outputs = [gr.Text(label='Name of the dish'), gr.JSON(label="Ingredients with corresponding quantities"), gr.Textbox(label="Instructions to prepare")]
    generate_btn.click(fn=smart_chef, inputs=inputs, outputs=outputs)

if __name__=="__main__":
    demo.launch(share=True)
```

In the terminal, run the following command
```console
python src/app.py
```

**Deploying Gradio application in HuggingFace Spaces**
* Create a HuggingFace account
* Install Gitbash (Optional)