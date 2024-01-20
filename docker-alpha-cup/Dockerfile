FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

COPY requirements.txt .
COPY tokenizer ./tokenizer
COPY model_brand9 ./model_brand
COPY model_good5 ./model_good

RUN apt update \
    && apt install --no-install-recommends -y python3 python3-pip \
    && pip install --upgrade pip \
    && python3 -m pip install -r requirements.txt
