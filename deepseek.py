# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

client = OpenAI(api_key="sk-11caf0a3051f498b9029e2af3ca7e23c", base_url="https://api.deepseek.com")

start = True
round = 0
reasoning_content = ""
content = ""
messages = []
while start:
    userInput = input("user: ")
    if userInput == '退出':
        break
    if round >= 2:
        messages.append({"role": "assistant", "content": content})
    messages.append({"role": "user", "content": userInput})
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        stream=True
    )

    messages.append(response.choices[0].message)
    print(f"Messages Round 1: {messages}")
    print(response.choices[0].message.content)

    for chunk in response:
        if chunk.choices[0].delta.reasoning_content:
            reasoning_content += chunk.choices[0].delta.reasoning_content
        else:
            content += chunk.choices[0].delta.content
    round += 1
