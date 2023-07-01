FROM nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.10 python3-pip python3-setuptools python3-distutils

WORKDIR /neiro-2023

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/neiro-2023/"

COPY . .
CMD ["python3", "llm_chat/chat.py"]
