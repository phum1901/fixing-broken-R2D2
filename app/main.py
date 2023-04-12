import argparse

import streamlit as st

from src.stage_model import StageModel


def main(args):
    model = StageModel(args.model_path)

    st.title("Broken R2D2")
    # inputs = st.text_input("Type something...")
    inputs = st.text_area("Type something...", height=150)
    if st.button("Generate Text"):
        response = model.predict(inputs)
        st.write(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model.pt")
    args = parser.parse_args()
    main(args)
