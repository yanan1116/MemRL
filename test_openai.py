import os
from openai import OpenAI

# openai
# client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])


for port in [1700, 1701]:
  client = OpenAI(
      base_url=f"http://10.225.68.16:{port}/v1",# localhost
      api_key="yyy",
  )

  response = client.chat.completions.create(
            model =  'Qwen/Qwen3-30B-A3B-Instruct-2507-FP8', 
            #response_format={ "type": "json_object" },
            messages=[
              {"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": 'tell me about the history of toronto, ontario'}
            ], max_tokens=1024, temperature=0
          )
  output = response.choices[0].message.content.strip()
  print(output )
  print('-'*20)






