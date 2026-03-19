import os
from openai import OpenAI

client = OpenAI(api_key = os.getenv("EVAL_OPENAI_API_KEY"), base_url = os.getenv("EVAL_OPENAI_BASE_URL"))
model = "gpt-4.1-mini"



messages = [
   {
       "role": "user",
       "content": "Reply with the single word: test"
   }
]


response = client.chat.completions.create(
    model = model,
    messages = messages,
    max_tokens = 5
)
print("Model results (Should be 'test'): " + str(response.choices[0].message.content))

