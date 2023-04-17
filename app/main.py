import argparse
import json

import requests
import streamlit as st

from src.stage_model import StageModel


def main(args):
    predictor = PredictorBackend(args.model_url)

    st.title("Broken R2D2")
    max_tokens = st.slider("Max characters to generate", 0, 10000, step=100, value=100)
    temperature = st.slider(
        "Select temperature",
        0.1,
        5.0,
        step=0.1,
        value=1.0,
        help="Medium temperature (0.3 to 0.7): Balanced creativity and coherence. High temperature (0.7 to 1): Highly creative and diverse, but potentially less coherent",
    )
    top_k = st.slider(
        "Select top K candidates",
        1,
        100,
        step=1,
        value=20,
        help="Consider the top k characters with the highest probability of occurring next",
    )

    # inputs = st.text_input("Type something...")
    inputs = st.text_area("Type something...", height=150)
    if st.button("Generate Text") and len(inputs) > 0:
        response = predictor.run(inputs, max_tokens=max_tokens, temperature=temperature, top_k=top_k)
        st.write(response)


class PredictorBackend:
    def __init__(self, url) -> None:
        if url is not None:
            self.url = url
            self._generate = self._generate_from_endpoint
        else:
            model = StageModel()
            self._generate = model.generate

    def run(self, text, max_tokens, temperature=1.0, top_k=None):
        output = self._generate(text, max_tokens, temperature, top_k)
        return output

    def _generate_from_endpoint(self, text, max_tokens, temperature, top_k):
        headers = {"Content-type": "application/json"}
        payload = {
            "text": text,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
        }
        payload = json.dumps(payload)

        response = requests.post(self.url, data=payload, headers=headers)
        print(response.json())
        output = response.json()["output"]
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_url", type=str, default=None)
    args = parser.parse_args()
    main(args)
