import os
from openai import OpenAI
from dotenv import load_dotenv
from KnowledgeRetrieval.config import *

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def run_llm(prompt):
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    try:
        completion = client.chat.completions.create(
            model=MODELNAME,
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            # temperature=0.7,
            timeout=30
        )
        result = completion.choices[0].message.content

        return result
    except Exception as e:
        print(e)
        return ""

if __name__ == '__main__':
    result = run_llm("hello?")
    print(result)