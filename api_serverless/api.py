import json

from src.stage_model import StageModel

model = StageModel()


def handler(event, context):
    text = event["text"]
    max_tokens = event["max_tokens"]
    temperature = event["temperature"]
    top_k = event["top_k"]

    output = model.generate(text, max_tokens, temperature, top_k)
    return {"output": output}
