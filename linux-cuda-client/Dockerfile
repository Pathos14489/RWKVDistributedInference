FROM python:3.10.6-slim-bullseye

WORKDIR /usr/src/app

COPY . .

RUN pip install --upgrade pip

RUN pip install torch numpy tqdm requests transformers

RUN apt update && apt install -y wget

RUN wget https://huggingface.co/BlinkDL/rwkv-4-pile-3b/resolve/main/RWKV-4-Pile-3B-20221008-8023.pth

CMD [ "python", "./client.py" ]