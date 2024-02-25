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
    # ingredients = [f"{item} ({quantity})" for item, quantity in ingredients.items()]
    return dish_name, ingredients, instructions

# Building interface
# demo = gr.Interface(
#     fn = smart_chef,
#     inputs=[gr.Textbox(label='Enter the list of ingredients you have, in a comma separated list')],
#     outputs=[gr.Text(label='Name of the dish'), gr.JSON(label="Ingredients with corresponding quantities"), gr.Textbox(label="Instructions to prepare")],
#     # allow_flagging='never'
# )

with gr.Blocks() as demo:
    gr.HTML("<h1 align='center'>Smart Chef</h1>")
    gr.HTML("<h3 align='center'><i>Cook with whatever you have</i></h3>")
    # gr.HTML("## Cook with whatever you have")
    inputs = [gr.Textbox(label='Enter the list of ingredients you have, in a comma separated text', lines=3, placeholder='Example: Chicken, Onion, Tomatoes, ... etc.')]
    generate_btn = gr.Button(value="Generate")
    outputs = [gr.Text(label='Name of the dish'), gr.JSON(label="Ingredients with corresponding quantities"), gr.Textbox(label="Instructions to prepare")]
    generate_btn.click(fn=smart_chef, inputs=inputs, outputs=outputs)

if __name__=="__main__":
    # demo.launch()
    demo.launch(share=True)