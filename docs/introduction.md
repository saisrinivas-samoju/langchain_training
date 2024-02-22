# [Langchain Introduction](https://python.langchain.com/docs/get_started/introduction)

Langchain is a framework for developing applications with Large Language Models.

## Why Langchain?

It provides a standard interface for developing applications with any LLM. Langchain does this by using its modules.

## [Modules](https://python.langchain.com/docs/modules/)
* Model I/O
* Retreival
* Chains
* Memory
* Agents

### [Model I/O](https://python.langchain.com/docs/modules/model_io/)
* Basic LLM Inputs and Outputs.
* More flexible than the existing model specific api formats.
* Learn the input and output formats for two type of models:
    * Text Completion models
    * Chat models

### [Retreival](https://python.langchain.com/docs/modules/data_connection/)
* Helps in connecting the LLM to a data source (like a vector database).
* Works in a standardized structure (that means, LLMs and Vector Databases can easily be swapped).
* Process: VectorDB > Query ways > Results to LLM

### [Chains](https://python.langchain.com/docs/modules/chains)
* For linking output of one model to another model.

### [Memory](https://python.langchain.com/docs/modules/memory/)
* For retaining the historical memory of our previous interactions in our own defined way.

### [Agents](https://python.langchain.com/docs/modules/agents/)
* Use the LLMs to choose the sequence of actions to take to perform a task using the tools we provide.