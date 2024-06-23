from traceloop.sdk import Traceloop

Traceloop.init(
    app_name="tangou1", api_key="781e7c1cde293be5d76469f35564745fc7d5443b4dd9d60e7327ef2de78b55262fd582aef075ab4029acff8ac42e7aa2", disable_batch=True
)


from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:4321/v1", api_key="lm-studio")

completion = client.chat.completions.create(
    model="Qwen/Qwen1.5-0.5B-Chat-GGUF",
    messages=[{"role": "system", "content": "Always answer in rhymes."}, {"role": "user", "content": "Introduce yourself."}],
    temperature=0.7,
)

print(completion.choices[0].message)