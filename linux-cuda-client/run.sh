#!/bin/bash

# install python3
sudo apt-get install python3 python3-pip wget

# pip install requirements
pip install torch torchdynamo numpy tqdm

if test -f "./RWKV-4-Pile-3B-20221008-8023.pth"; then
    echo "model exists"
    x-terminal-emulator -e python3 ./client.py
else
    echo "model not exists"
    x-terminal-emulator -e "wget https://huggingface.co/BlinkDL/rwkv-4-pile-3b/resolve/main/RWKV-4-Pile-3B-20221008-8023.pth && python3 ./client.py"
fi
