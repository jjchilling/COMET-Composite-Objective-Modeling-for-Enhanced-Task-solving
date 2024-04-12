from openai import OpenAI

client = OpenAI()

with open('point_robot.py', 'r') as file:
    pr_code = file.read()

# create a chat completion
chat_completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Write code for a detailed reward function for RL Project in Gymnax that makes the agent fly in a rectangle for PointRobot environment content:" + pr_code}])

# print the chat completion
print(chat_completion.choices[0].message.content)