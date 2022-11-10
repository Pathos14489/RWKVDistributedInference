########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM                                         #
########################################################################################################

import gc
from math import floor
import numpy as np
import os
import time
import types
import torch
from src.utils import TOKENIZER
import requests
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)


########################################################################################################
# User Config                                                                                          #
########################################################################################################

# Set this to the url of the work server that will be used to get jobs
WORK_SERVER_URL = "http://localhost:3000"

# Set by the work server on startup, don't worry about this unless you want to set a static ID for your client, 
# which will need to be registered with the work server first in order to work. 
# If you don't know what this is or why you might need/want it, don't worry about it.
SESSION_ID = None

# default 0.8 - Set yourself if you want to change it, you might be able to find a better value than this to get better results
TEMPERATURE = 0.8

# default 0.8 - Set yourself if you want to change it, you might be able to find a better value than this to get better results
TOP_P = 0.8

# you can try changing this, according to BlinkDL, this can be set to anywhere up to 2048 and remain coherent. 
# But it will be linearly slower(I think?) the higher you set it, so you might want to keep it at 1024.
CTX_LEN = 1024 

# This is the model being, you shouldn't need to change this unless you want to use a custom model
MODEL_NAME = './RWKV-4-Pile-3B-20221008-8023'

########################################################################################################

# makes sure there is a log file
if not os.path.exists("log"):
    with open("log", "w") as f:
        pass

def log(text):
    if "\n" in text:
        text = text.split("\n")
    else:
        text = [text]
    with open("log", "a") as f:
        for line in text:
            print(f"[{time.strftime('%H:%M:%S')}][{time.time()}] {line}")
            f.write(f"[{time.strftime('%H:%M:%S')}][{time.time()}] {line}\n")
            
def session(): # Gets session ID from work server, (or uses the one you set above) and checks if the server is alive
    global SESSION_ID
    if SESSION_ID == None:
        try:
            SESSION_ID = requests.get(WORK_SERVER_URL + "/api/session").json()["id"]
        except:
            log("Could not connect to work server! Contact the developer on Discord(https://discord.gg/FZyGH3PGfs), or try again later.")
            log("Exiting...")
            exit(1)
        with open("sessions", "a") as f:
            f.write(SESSION_ID + "\n")
        log("Got Session ID From Server: " + SESSION_ID)
    else:
        log("Session ID Manually Set: " + SESSION_ID)
        try:
            requests.get(WORK_SERVER_URL + "/api/status")
        except:
            log("Could not connect to work server! Contact the developer on Discord(https://discord.gg/FZyGH3PGfs), or try again later.")
            log("Exiting...")
            exit(1)
    return SESSION_ID 

########################################################################################################

args = types.SimpleNamespace()

# detect available devices
args.RUN_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# checks if the GPU has enough memory
if args.RUN_DEVICE == 'cuda':
    available_vram = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
    if torch.cuda.get_device_properties(0).total_memory >= 6 * 1024 * 1024 * 1024 and available_vram >= 6 * 1024 * 1024 * 1024:
        log('GPU memory is enough! Size is: ' + str(floor(torch.cuda.get_device_properties(0).total_memory/1024/1024/1024)) + 'GB\nUsing GPU')
    else:
        log('GPU memory is not enough! Must be at least 6GB.\nSize is: ' + str(torch.cuda.get_device_properties(0).total_memory/1024/1024/1024) + 'GB\nUsing CPU')
        args.RUN_DEVICE = 'cpu'
        
# fp32 // bf16 (saves VRAM, slightly less accurate) -- Should be fine to leave as is unless you're just curious, 
# or if your hardware doesn't support bf16
args.FLOAT_MODE = "bf16"

args.MODEL_NAME = MODEL_NAME
args.n_layer = 32 # no touchy, specific to model(I think)
args.n_embd = 2560 # no touchy, specific to model(I think)
args.ctx_len = CTX_LEN
args.vocab_size = 50277 # you shouldn't need to change this
args.head_qk = 0
args.pre_ffn = 0
args.grad_cp = 0
args.my_pos_emb = 0
os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE

init_state = None
init_out = None
state = None
out = None

########################################################################################################
    
def prompt_to_token_prompt(text):
    tokens = tokenizer.tokenizer.encode(text)
    return tokens
def token_prompt_to_text(tokens):
    text = tokenizer.tokenizer.decode(tokens)
    return text

def generate(prompt, temperature=0.8, top_p=0.8, max_len=128, stop = ["\n"]):
    global init_state, init_out, state, out
    ctx = prompt[:CTX_LEN]
    src_len = len(ctx)
    log('\nYour prompt now has ' + str(src_len) + ' tokens after trimming.\n************************************')
    print(token_prompt_to_text(ctx), end='')
    src_ctx = ctx.copy()
    
    ctx = src_ctx.copy()
    # initialize variables
    for i in range(src_len):
        x = ctx[: i + 1]
        if i == src_len - 1:
            init_out, init_state = model.forward(x, init_state)
        else:
            init_state = model.forward(x, init_state, preprocess_only=True)
    gc.collect()
    torch.cuda.empty_cache()
        
    for i in range(src_len, src_len + max_len):
        x = ctx[: i + 1]
        x = x[-CTX_LEN:]

        if i == src_len:
            out = init_out.clone()
            state = init_state.clone()
        else:
            out, state = model.forward(x, state)
            
        out[0] = -999999999  # disable <|endoftext|>

        ttt = tokenizer.sample_logits(
            out,
            x,
            CTX_LEN,
            temperature=temperature,
            top_p_usual=top_p
        )
        string = tokenizer.tokenizer.decode([int(ttt)])
        print(string, end='', flush=True)
        apend = True
        for s in stop:
            if s in string: 
                apend = False
        if apend:
            ctx += [ttt]
        else:
            break
    ctx = ctx[src_len:]
    ctx = [c.item() for c in ctx]
    return ctx

########################################################################################################
            
log("Starting...")

session() # test server connection and get session id
    
from src.model_run import RWKV_RNN

model_type = 'RWKV' # 'RWKV' or 'RWKV-ffnPre'

pre_usage = torch.cuda.memory_allocated() / 1024 / 1024 # GPU memory usage before loading model
model = RWKV_RNN(args)
post_usage = torch.cuda.memory_allocated() / 1024 / 1024 # GPU memory usage after loading model
vram_usage = (post_usage - pre_usage) / 1024 # GPU memory usage in GB
log(f"Using {vram_usage}GB of VRAM") 

tokenizer = TOKENIZER(["20B_tokenizer.json","20B_tokenizer.json"], UNKNOWN_CHAR=None) # 20B_tokenizer.json is the default tokenizer, you shouldn't need to change this

########################################################################################################

# get batch from /api/batch

def get_batch():
    global SESSION_ID
    if SESSION_ID == None:
        session()
    r = requests.get(WORK_SERVER_URL + "/api/batch?session=" + SESSION_ID)
    return r.json()

# for each job in the batch, finish the caption with generate() and send it back to /api/job/:id

def send_job(job):
    job_id = job["id"]
    prompt = job["tokenized"]
    # log("Generating caption for job " + str(job_id) + "...")
    st = time.time()
    caption = generate(prompt)
    en = time.time()
    
    log("Generated caption for job " + str(job_id) + " in " + str(round(en - st, 2)) + "s")
    
    requests.post(WORK_SERVER_URL + "/api/job/" + str(job_id), json={
        "session": SESSION_ID,
        "id": job_id,
        "tokens": caption,
        "start": st,
        "end": en,
        "time": en - st,
    })

########################################################################################################

# main loop
    
x = 0
jobs = []
while True:
    x+=1
    jobs = get_batch()
    if len(jobs) > 0:
        x = 0
        for job in tqdm(jobs):
            log(job)
            send_job(job)
    if x > 5:
        print("IDLE")
        time.sleep(60)