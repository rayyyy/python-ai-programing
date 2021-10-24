FROM tensorflow/tensorflow:latest-gpu

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN apt-get update && \
  apt-get install -y git

RUN pip install jupyterlab

WORKDIR /app
COPY . /app