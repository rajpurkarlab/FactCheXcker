# ============================ #
# IMPORTS                      #
# ============================ #

import openai
from openai import AzureOpenAI
from openai import OpenAI
import os
from constants import OPENAI_API_VERSION, OPENAI_AZURE_ENDPOINT, OPENAI_MODEL_NAME

# ============================ #
# LLM Client (uncomment/add)   #
# ============================ #

# Default OpenAI (uncomment if )
# client = AzureOpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
# )

# Azure OpenAI
client = AzureOpenAI(
    api_version=OPENAI_API_VERSION,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=OPENAI_AZURE_ENDPOINT,
)


def run_query(system, user):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME, messages=messages, max_tokens=500
        )
        content = response.choices[0].message.content
        content = content.replace("'", '"')
        return content, response
    except Exception as e:
        print(f"Error: {e}")
