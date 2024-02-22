# Project Setup

## Virtual environment creation
### Conda environment
Global environment
```bash
conda create -n env_name
```
Local environment (considering the local environment to be the current directory)
```bash
conda create -p ./env_name
```
### Python virtual environment
```bash
python -m venv env_name
```
Local environment
```bash
python -m venv ./env_name
```

## Activating virtual environment
### Conda environment
Global environment
```bash
conda activate env_name
```
Local environment
```bash
conda activate ./env_name
```
### Python virtual environment
```bash
source env_name/Scripts/activate
```
Local environment
```bash
source ./env_name/Scripts/activate
```

## Required packages
* langchain
* langchain-core
* langchain-community
* langchain-experimentation
* openai
* llama_cpp_python

## Install requirements.txt file
Prepare the requirements.txt file with the above package list
```bash
pip install -r requirements.txt
```