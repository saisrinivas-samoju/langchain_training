import os
from typing import List
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
import gradio as gr

# Creating the instance of the chat model
with open('openai_api_key.txt') as f:
    os.environ['OPENAI_API_KEY'] = f.read()

chat = ChatOpenAI()

class SmartChef(BaseModel):
    name: str = Field(description='Name of the dish')
    ingredients: dict = Field(description='Python dictionary of ingredients and their corresponding quantities as key and values of the python dictionary respectively.')
    instructions: List[str] = Field(description='Python list of instructions to prepare the dish')
    
output_parser = PydanticOutputParser(pydantic_object=SmartChef)


def smart_chef(food_items: str):
    human_template = """I have the following list of food items:

    {food_items}

    Suggest me a recipe only using these food items

    {format_instructions}
    """

    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
    prompt = chat_prompt.format_prompt(
        food_items=food_items,
        format_instructions=output_parser.get_format_instructions()
    )

    messages = prompt.to_messages()

    response = chat(messages=messages)

    output = output_parser.parse(response.content)
    
    dish_name = output.name
    ingredients = output.ingredients
    instructions = output.instructions

    return (dish_name, ingredients, instructions)

demo = gr.Interface(
    fn=smart_chef,
    inputs=[
        gr.Textbox(label='Enter the list of ingredients you have', lines=3, placeholder="Example: Rice, Chicken ... etc")
    ],
    outputs=[
        gr.Text(label='Nmae of the dish'),
        gr.JSON(label="Ingredients required with respective quantitites"),
        gr.Textbox(label='Instructions')
    ],
    allow_flagging='never'
)

demo.launch(share=True)
