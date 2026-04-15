import os
from openai import OpenAI,AzureOpenAI

# openai
# client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# azure
client = AzureOpenAI(
      azure_endpoint = "https://sub-lgeci-tai-opneai.openai.azure.com/", 
      # api_key=os.environ['AZURE_OPENAI_API_KEY'],  
      # api_version= "2024-10-01-preview",
      api_version="2024-03-01-preview"
    )

response = client.chat.completions.create(
          model =  'gpt-5-mini', 
          #response_format={ "type": "json_object" },
          messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": 'tell me about the history of toronto, ontario'}
          ], max_completion_tokens=30
        )
output = response.choices[0].message.content.strip()
print('response--->', output)
print(response.usage.prompt_tokens)
print(response.usage.completion_tokens)
print(response.usage.total_tokens)


print('\nembedding -->')

for model in ["text-embedding-3-large", "text-embedding-3-small"]:
    resp = client.embeddings.create(
        model= model, 
        input="tell me about the history of toronto, ontario"
    )

    print(model)
    print(len(resp.data))
    print(len(resp.data[0].embedding))

print('-'*5)
resp = client.embeddings.create(
  model="text-embedding-3-large",
  input=["hello world", "toronto history"]
)

print(len(resp.data))
print(len(resp.data[0].embedding))