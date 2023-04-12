FROM python:3.9.16-buster

# install dependencies 
COPY requirements/prod.txt ./requirements.txt
RUN pip install --upgrade pip==23.0.1
RUN pip install -r requirements.txt
ENV PYTHONPATH ".:"

#
COPY src/ ./src
COPY app/ ./app 
COPY model.pt model.pt 

ENTRYPOINT ["streamlit", "run", "app/main.py", "--", "--model_path", "model.pt"]