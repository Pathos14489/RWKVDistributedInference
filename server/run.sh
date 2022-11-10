#!/bin/bash

# install python3
sudo apt-get install python3 python3-pip

# pip install requirements
pip install transformers flask
x-terminal-emulator -e python3 ./client.py
