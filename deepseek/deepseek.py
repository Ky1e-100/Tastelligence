import requests
import dotenv
import os
import json


def requestSMILES(ingredient):
    dotenv.load_dotenv()

    API_KEY = os.getenv("API_KEY")
    API_URL = 'https://openrouter.ai/api/v1/chat/completions'

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    data = {
        "model": "deepseek/deepseek-chat:free",
        "messages": [{"role": "user", "content": f"What is the 3 most common molecular SMILE contained in {ingredient}? Give it to me in a list with only name of compound and chemical SMILE, no extra words"}]
    }

    response = requests.post(API_URL, json=data, headers=headers)

    if response.status_code == 200:
        data = response.json()
        content = data['choices'][0]['message']['content']
        return content
    else:
        print("Failed to fetch data from API. Status Code:", response.status_code)
        
print(requestSMILES("soya sauce"))