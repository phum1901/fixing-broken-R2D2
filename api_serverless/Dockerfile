FROM public.ecr.aws/lambda/python:3.9

COPY requirements/prod.txt ./requirements.txt
RUN pip install --upgrade pip==23.0.1
RUN pip install -r requirements.txt

COPY src/ ./src
COPY api_serverless/api.py ./api.py
COPY artifacts/model.pt artifacts/model.pt

CMD ["api.handler"]
