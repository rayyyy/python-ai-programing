FROM python:3.9

COPY requirements.txt .
RUN pip3 install --upgrade pip && \
  pip3 install -r requirements.txt && \
  apt-get update && \
  apt-get install -y git

RUN pip install jupyterlab

WORKDIR /app
COPY . /app