FROM python:3.9-slim

# install dependencies
COPY requirements/prod.txt ./requirements.txt
RUN pip install --upgrade pip==23.0.1
RUN pip install -r requirements.txt
ENV PYTHONPATH ".:"

#
COPY src/ ./src
COPY app/ ./app
COPY artifacts/model.pt artifacts/model.pt

EXPOSE 8501

# to use serverless model backend add the line below to ENTRYPOINT
# "--", "--model_url", "REPLACE_THIS_WITH_YOUR_URL"
ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8501"]