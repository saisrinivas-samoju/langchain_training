# Python Basics

Questions:
* What all data structures take position in count? (list, tuple, string does) (dictionary and set doesn't)
* What are *args & **kwargs?


*args -> For any number of positional arguments
```py
def get_prod(*args):
    res = 1
    for arg in args:
        res = res*arg
    return res

get_prod(2, 3, 4, 5)
```
    > 120

**kwargs -> For any number of keywrod arguments

```py
def greet(**kwargs):
    greeting = "Hello"
    if 'name' in kwargs:
        greeting += f", {kwargs['name']}"
    if 'age' in kwargs:
        greeting += f", you are {kwargs['age']} years old"
    if 'location' in kwargs:
        greeting += f" from {kwargs['location']}"
    greeting+="!"
    return greeting

print(greet(name="John"))
print(greet(name="John", age=24))
print(greet(name="John", location='New York'))
print(greet(name="John", age=24, location='New York'))
```
    Hello, John!
    Hello, John, you are 24 years old!
    Hello, John from New York!
    Hello, John, you are 24 years old from New York!

```py
def arg_test(*args):
    return args

arg_test(1, 2, 3)
```
    > (1, 2, 3)


```py
def kwarg_test(**kwargs):
    return kwargs

kwarg_test(a=1, b=3)
```
    > {'a': 1, 'b': 3}




```python
# Always args should be followed by kwargs
# args/positional arguments are read in the form of a tuple (where position matters)
# While kwargs/keyword arguments are read in dictionaries (where positions doesn't matter)
```


```python
# Decorators
import time
from datetime import datetime

get_curr_time = lambda :datetime.now().time().strftime("%H:%M:%S")

def timer(func):
    def get_timings(*args, **kwargs): # Wrapper function
        start_time = get_curr_time()
        val = func(*args, **kwargs)
        time.sleep(5)
        end_time = get_curr_time()
        print(f"Start time: {start_time} | End time: {end_time} ")
        return val
    return get_timings

@timer
def get_sq(a):
    return a**2
```


```python
get_sq(4)
```

    Start time: 14:56:15 | End time: 14:56:20 

    16



# Chains

* Chains allows us to connect one LLM response to another.


```python
import os
from dotenv import load_dotenv
from langchain_openai.llms import OpenAI
from langchain_openai.chat_models import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate

load_dotenv()
llm = OpenAI()
chat = ChatOpenAI()
set_llm_cache(InMemoryCache())
```

## LLMChain

* Basic Building Block of the Chains..
* Simple LLM call with an input and output.
* This is not exactly a chain but a building block of it.


```python
# Previously, we created functions to take the input from the use for the prompt template and get the response using the LLM
# Using LLM Chain, this becomes easy

human_template = "Write a film story outline on the topic: {topic}"

human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
```


```python
from langchain.chains import LLMChain

chain = LLMChain(llm=chat, prompt=chat_prompt) # Chain with one block
```


```python
# For making a call

result = chain.invoke(input={"topic": "Growth of India"})
result
```

    {'topic': 'Growth of India',
     'text': "Title: Rising India\n\nAct 1:\n- The film opens with a montage of historical footage ... to shape a better future for themselves and generations to come."}




```python
# How to do it LCEL

lcel_chain = chat_prompt | chat # same as chat_prompt -> chat -> get the response

lcel_chain.invoke(input={"topic": "Growth of India"})
```

    AIMessage(content="Title: Rising India\n\nAct 1:\n- The film opens with a montage of historical footage ... to shape a better future for themselves and generations to come.")



## Simple Sequential Chain

The blocks in the simple sequential chain produce a single output


```python
story_line_template = "Write a film story outline on the topic: {topic}"
story_line_prompt = ChatPromptTemplate.from_template(template=story_line_template)
story_line_chain = LLMChain(llm=chat, prompt=story_line_prompt)
```


```python
full_story_template = "Write a short film story on the given story outline: {story_line}"
full_story_prompt = ChatPromptTemplate.from_template(template=full_story_template)
full_story_chain = LLMChain(llm=chat, prompt=full_story_prompt)
```


```python
reviewer_template = "Act as a reviewer in rotten tomatoes and rate the given story: {full_story}"
reviewer_prompt = ChatPromptTemplate.from_template(template=reviewer_template)
reviewer_chain = LLMChain(llm=chat, prompt=reviewer_prompt)
```


```python
from langchain.chains import SimpleSequentialChain

film_chain = SimpleSequentialChain(
    chains = [story_line_chain, full_story_chain, reviewer_chain], # In order, the intermediate input and ouptuts acts as 'args' in python.
    verbose=True # to look at the internal response
)
```


```python
result = film_chain.run(input={"topic": "Growth of India"})
result
```

    
    
    [1m> Entering new SimpleSequentialChain chain...[0m
    
    
    [1m> Entering new LLMChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mHuman: Write a film story outline on the topic: Growth of India[0m
    
    [1m> Finished chain.[0m
    [36;1m[1;3mTitle: Rising India
    
    Act 1:
    - The film opens ...
    
    Act 2:
    - As Ravi's career progresses, ...
    
    Act 3:
    - The tension between Ravi and ...
    
    Rising India is a story of resilience,... to come.[0m
    
    ...

    [1m> Finished chain.[0m
    

    'As a reviewer on Rotten Tomatoes, I would rate "Rising India" with a solid 4 out of 5 stars. ... circumstances, and a testament to the power of individuals to make a difference.'




```python
# Let's try it with the existing LLMChain blocks and LCEL (This won't work)

lcel_film_chain = story_line_chain | full_story_chain | reviewer_chain

lcel_film_chain.invoke(input={"topic": "Growth of India"})


# This won't work as the output from one chain is not targetted to the input in another chain with the same key/variable name in LCEL.
# As the intermediate input and output in LCEL only takes as 'kwargs' in python
# To do that, we need to use something called as RunnableLambda class that is accepted by the LCEL chains
# Let's see how below
```

    
    
    [1m> Entering new LLMChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mHuman: Write a film story outline on the topic: Growth of India[0m
    
    [1m> Finished chain.[0m
    
    
    [1m> Entering new LLMChain chain...[0m
    

    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[77], line 5
          1 # Let's try it with the existing LLMChain blocks and LCEL (This won't work)
          3 lcel_film_chain = story_line_chain | full_story_chain | reviewer_chain
    ----> 5 lcel_film_chain.invoke(input={"topic": "Growth of India"})

    ValueError: Missing some input keys: {'story_line'}



```python
full_story_chain.invoke(input={"story_line": "Growth of India"})
```

    
    
    [1m> Entering new LLMChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mHuman: Write a short film story on the given story outline: Growth of India[0m
    
    [1m> Finished chain.[0m
    

    > {'story_line': 'Growth of India',
     'text': "Title: Rising India\n\nIn a small village in rural India, a group of children play cricket .... India is rising, and nothing can stop its upward trajectory."}




```python
# Using Expression Language with the existing LLMChain blocks

from langchain.schema.runnable import RunnableLambda

def get_story_line(response):
    return {"story_line": response['text']}

def get_full_story(response):
    return {"full_story": response['text']}

story_line_lcel_chain = story_line_chain | RunnableLambda(get_story_line)
full_story_lcel_chain = full_story_chain | RunnableLambda(get_full_story)

lcel_film_chain = story_line_lcel_chain | full_story_lcel_chain | reviewer_chain

lcel_film_chain.invoke(input={"topic": "Growth of India"})
```

    
    
    [1m> Entering new LLMChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mHuman: Write a film story outline on the topic: Growth of India[0m
    ...

    > {'full_story': 'The film "Rising India" follows the journey of Ravi, a young man ... unity, and the unstoppable rise of a nation on the brink of greatness.',
     'text': 'As a reviewer on Rotten Tomatoes, ... of individuals to make a difference.'}


```python
# Completly using LCEL
from langchain.schema.runnable import RunnableLambda

def get_story_line(ai_message):
    return {"story_line": ai_message.content}

def get_full_story(ai_message):
    return {"full_story": ai_message.content}


story_line_template = "Write a film story outline on the topic: {topic}"
story_line_prompt = ChatPromptTemplate.from_template(template=story_line_template)
story_line_chain = story_line_prompt | chat | RunnableLambda(get_story_line)

full_story_template = "Write a short film story on the given story outline: {story_line}"
full_story_prompt = ChatPromptTemplate.from_template(template=full_story_template)
full_story_chain = full_story_prompt | chat | RunnableLambda(get_full_story)

reviewer_template = "Act as a reviewer in rotten tomatoes and rate the given story: {full_story}"
reviewer_prompt = ChatPromptTemplate.from_template(template=reviewer_template)
reviewer_chain = reviewer_prompt | chat

lcel_film_chain = story_line_chain | full_story_chain | reviewer_chain

lcel_film_chain.invoke(input={"topic": "Growth of India"})

# from langchain.callbacks.tracers import ConsoleCallbackHandler # for verbose
# lcel_film_chain.invoke(input={"topic": "Growth of India"}, config={'callbacks': [ConsoleCallbackHandler()]})
```
    AIMessage(content='As a reviewer on Rotten Tomatoes, I would rate "Rising India" with a solid 4 out of 5 stars. ... circumstances, and a testament to the power of individuals to make a difference.')



## Sequential Chain

Similar to Simple Sequential Chains but it allows us to access all the intermediate outputs.


```python
story_line_template = "Write a film story outline on the topic: {topic}"
story_line_prompt = ChatPromptTemplate.from_template(template=story_line_template)
story_line_chain = LLMChain(llm=chat, prompt=story_line_prompt, output_key="story_line")

full_story_template = "Write a short film story on the given story outline: {story_line}"
full_story_prompt = ChatPromptTemplate.from_template(template=full_story_template)
full_story_chain = LLMChain(llm=chat, prompt=full_story_prompt, output_key="full_story")

reviewer_template = "Act as a reviewer in rotten tomatoes and rate the given story: {full_story}"
reviewer_prompt = ChatPromptTemplate.from_template(template=reviewer_template)
reviewer_chain = LLMChain(llm=chat, prompt=reviewer_prompt, output_key="reviewer_response")
```


```python
from langchain.chains import SequentialChain

seq_chain = SequentialChain(
    chains = [story_line_chain, full_story_chain, reviewer_chain],
    input_variables=['topic'],
    output_variables=['story_line', 'full_story', 'reviewer_response'],
    verbose=True, # doesn't matter here, as we get full output from each block with SequentialChain.
)

seq_chain.invoke(input={'topic': "Growth of India"})
```
    
    [1m> Entering new SequentialChain chain...[0m
    
    
    [1m> Entering new LLMChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mHuman: Write a film story outline on the topic: Growth of India[0m
    
    Act 1:
    - The film opens with a montage of historical footage showcasing India's struggle for independence and ...
    
    Act 2:
    - As Ravi's career progresses, we see the economic and social changes happening in India. ...
    
    Act 3:
    - The tension between Ravi and the powerful forces trying to silence him comes to a head....
    
    Rising India is a story of resilience, hope, and the power of ordinary people to ... come.[0m


    > {'topic': 'Growth of India',
     'story_line': "Title: Rising India\n\nAct 1:\n- The film opens with a montage of historical footage ... individuals to make a difference.'}




```python
# Using LCEL with the existing LLMChain objects (This will work)

lcel_film_chain = story_line_chain | full_story_chain | reviewer_chain

lcel_film_chain.invoke(input={"topic": "Growth of India"})
```

    
    
    [1m> Entering new LLMChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mHuman: Write a film story outline on the topic: Growth of India[0m
    
    [1m> Finished chain.[0m
    
    
    [1m> Entering new LLMChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mHuman: Write a short film story on the given story outline: Title: Rising India
    
    Act 1:
    - The film opens with a montage of ... communities.
    
    Act 2:
    - As Ravi's career progresses, ... power and profits.
    
    Act 3:
    - The tension between Ravi ... to come.[0m
    
    [1m> Finished chain.[0m
    

    > {'topic': 'Growth of India',
     'story_line': "Title: Rising India\n\nAct 1:\n- The film opens with a montage of historical footage showcasing India's ... testament to the power of individuals to make a difference.'}



## LCEL

What exactly is this LCEL and why the syntax is like this?

```python
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough, RunnableParallel

# RunnableLambda -> Takes a function -> Returns the output
## Can be added in chains

def square(a):
    return a**2

sq_runnable = RunnableLambda(square)

sq_runnable
```
    RunnableLambda(square)

```python
type(sq_runnable)
```

    langchain_core.runnables.base.RunnableLambda


```python
sq_runnable.invoke(4) # Chain object, so invoke would work
```
    16

```python
from langchain.schema.runnable import RunnablePassthrough

# RunnablePassthrough -> Takes the input -> Returns the input
## Use to pass the input from one chain block to another

runnable_sq_pass = RunnablePassthrough()
```

```python
runnable_sq_pass.invoke(4)
```
    4


```python
from langchain.schema.runnable import RunnableParallel

# RunnableParellel -> Runs multiple chain blocks at a time
@timer
def times2(x):
    return x*2
@timer
def times3(y):
    return y*3
@timer
def add(res_dict):
    return res_dict['a'] + res_dict['b']

runnable_times2 = RunnableLambda(times2)
runnable_times3 = RunnableLambda(times3)
par_chain = RunnableParallel({"a": runnable_times2, "b": runnable_times3}) # Runs parallelly
runnable_sum = RunnableLambda(add)

calc_chain = par_chain | runnable_sum
```

```python
# If you want to pass different values
from operator import itemgetter
par_chain = RunnableParallel(
    {
        "a": itemgetter("x") | runnable_times2, 
        "b": itemgetter("y") | runnable_times3
    }
)

calc_chain = par_chain | runnable_sum
```


```python
calc_chain.invoke(input={"x": 2, "y": 3})
```

    Start time: 01:39:04 | End time: 01:39:09 Start time: 01:39:04 | End time: 01:39:09 
    
    Start time: 01:39:09 | End time: 01:39:14 
    
    > 13

```python
# Let's use all the above runnables together

@timer
def sum_up(res_dict):
    res = 0
    for k, v in res_dict.items():
        res+=v
    return res

runnable_sum_up = RunnableLambda(sum_up)

par_chain2 = RunnableParallel(
    {
        "a": itemgetter("x") | runnable_times2, 
        "b": itemgetter("y") | runnable_times3,
        "x": itemgetter("x") | RunnablePassthrough(),
        "y": itemgetter("y") | RunnablePassthrough()
    }
)

calc_chain = par_chain2 | runnable_sum_up

calc_chain.invoke(input={"x": 2, "y": 3})
```

    Start time: 01:43:45 | End time: 01:43:50 
    Start time: 01:43:45 | End time: 01:43:50 
    Start time: 01:43:50 | End time: 01:43:55 
    
    > 18




```python
# Now that we understood how to use the syntax, let's understand why the syntax is like this (with a pipe operator)

# Class
class RunnableLambdaTest:
    def __init__(self, func):
        self.func = func
    
    def __or__(self, other_runnable_obj):
        def chained_func(*args, **kwargs):
            return other_runnable_obj.invoke(*args, **kwargs)
        return RunnableLambdaTest(chained_func)
    
    def invoke(self, *args, **kwargs):
        return self.func(*args, **kwargs)

# Implementation
def times2(a):
    return a*2

def times3(a):
    return a*3

runnable_2 = RunnableLambdaTest(times2)
runnable_3 = RunnableLambdaTest(times3)

test_chain = runnable_2 | runnable_3

test_chain.invoke(2)
```
    > 6



## LLM Router Chain

LLMRouterChain can take in an input and redirect it to the most appropriate LLMChain Sequence.