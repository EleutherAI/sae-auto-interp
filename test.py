# %%
import os

os.environ["CONFIG_PATH"] = "configs/caden_gpt2.yaml"
from sae_auto_interp.clients import get_client

client = get_client('local', "casperhansen/llama-3-70b-instruct-awq")

client

# %%async

message = [
    {
        "role" : "system",
        "content" : "Format your response within a python array."
    },
    {
        "role" : "user",
        "content" : "Give me a integer between 0 and 1"
    },
]

result = await client.generate(
    message,
    temperature=0.0,
    max_tokens=2,
    # raw=True,
)

result

# %%

template = """{% raw %}
{% if messages[0]['role'] == 'system' %}
System: {{ messages[0]['content'] }}
{% endif %}

{% for message in messages[1:] %}
{% if message['role'] == 'user' %}
Human: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}
Assistant: {{ message['content'] }}
{% endif %}
{% endfor %}

{% if add_generation_prompt %}
Human: Can you tell me about {{ topic }}?
Assistant:
{% endif %}
{% endraw %}"""

result = await client.client.chat.completions.create(
  model="casperhansen/llama-3-70b-instruct-awq",
  messages=[
    {"role": "user", "content": "What is the weather like today?"}
  ],
  extra_body={
    # "guided_choice": ["sunny", "rainy", "cloudy"]
    "chat_template_kwargs":None,
  }
)

# %%

result.choices[0].message.content