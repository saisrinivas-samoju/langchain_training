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
ipykernel
langchain==0.0.351
langchain-community==0.0.4
langchain-core==0.1.1
langsmith==0.0.72
<!-- langchain-experimentation -->
openai==1.6.0
llama_cpp_python
mkdocs-material
pydantic==2.5.2
pydantic_core==2.14.5
gradio==4.19.2
gradio_client==0.10.1
langchain_openai==0.0.8

## Install requirements.txt file
Prepare the requirements.txt file with the above package list
```bash
pip install -r requirements.txt
```