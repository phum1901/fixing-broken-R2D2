# Fixing Broken R2D2
Educational project on the Transformer's encoder, trained on Star Wars English subtitle dataset at the character level with a simple app UI using Streamlit, as well as a serverless API using Lambda.
## Install

```bash
pip install --upgrade pip==23.0.1
pip install pip-tools
pip-compile requirements/prod.in && pip-compile requirements/dev.in
pip-sync requirements/prod.txt requirements/dev.txt
export PYTHONPATH=.
```

## Quick start

You can train a model for 100 epochs or trigger early stopping (configurable through config/from_scratch.yaml). While using wandb to track the experiments is optional.
```bash
python src/train.py --config config/from_scratch.yaml --wandb
```

After the training done export your model checkpoint to torch script for inference speed

```bash
python src/stage_model.py --ckpt_path $YOUR_CKPT_PATH --save_path $YOUR_SAVE_PATH
```
starting local serverless (optional)

```bash
docker build -f api_serverless/Dockerfile -t serverless:latest .
docker run -d -p 9000:8080 serverless:latest
```

starting streamlit app
```bash
streamlit run app/main.py
```

or using docker
```bash
docker build -f app/Dockerfile -t streamlit:latest .
docker run -d -p 8501:8501 streamlit:latest
```

