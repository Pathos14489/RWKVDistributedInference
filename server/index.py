import os
import json
import random
import uuid
import time
from flask import request, jsonify, Flask
from transformers import PreTrainedTokenizerFast

batch_size = 100 # number of jobs to send in a batch -- pull this number out of my ass, so try tuning it until it's just right for your latency and hardware.

tokenizer = PreTrainedTokenizerFast(tokenizer_file="20B_tokenizer.json")

work_being_done = [] # list of jobs that are currently being worked on by workers
active_workers = {} # list of active workers
job_cache = {} # caches jobs in memory -- faster than reading from disk every time if you don't want to run this on an SSD
job_dir = "./jobs/" # Where jobs prompts are stored -- every job exists solely as a prompt file, and the parameters are set by the workers to encourage them to strive for the best scores on their own. and because frankly, I'm not sure what the best parameters are for RWKV.
work_dir = "./work/" # Where work is stored after it's been done and returned to the server from a worker client.
minimum_work = 5 # noticed that in small amounts of work, it could sometimes doo over 5 but not always and not very many over typically. I think it's race condition related, but frankly it isn't really that big of a concern to me, as it only results in getting extra data so who gives a shit.

#Initialize the directory
os.makedirs(work_dir, exist_ok=True)
os.makedirs(job_dir, exist_ok=True)
os.makedirs("workers/", exist_ok=True)
if not os.path.exists("./verified"):
    open("./verified", 'w').close()

def create_api(): # create the Flask API object
    api = Flask(__name__)
    return api

api = create_api()

def verify_session(session): # verify that the session_id supplied is valid
    if session == None:
        print("No session")
        return False
    with open("./verified", 'r') as f:
        for line in f:
            if line.strip() == session:
                print("Verified session")
                # log ping to worker file
                with open(f"./workers/{session}", 'a') as f:
                    f.write(str(time.time()) + "\n")
                if session not in active_workers:
                    active_workers[session] = time.time()
                else:
                    if time.time() - active_workers[session] > 60 * 60 * 3:
                        del active_workers[session]
                return True
    print("Unverified session")
    return False
    
def get_job_cache(): # load all jobs into memory
    files = os.listdir(job_dir)
    for file in files:
        if file not in job_cache:
            job_prompt = open(job_dir + file, 'r').read()
            job = {
                "id": file.split(".")[0],
                "tokenized": tokenizer.encode(job_prompt, add_special_tokens=False),
            }
            job_cache[file.split(".")[0]] = job
            os.makedirs(work_dir + file.split(".")[0], exist_ok=True) # Create a directory for the job to store work in
            
@api.route('/api/status', methods=['GET'])
def status(): # return the status of the server as ok if it's running
    return "ok"
# DOCS: GET /api/status - returns ok - used to check if the server is running

def get_job_work(job_id): # get all work done for a given job id
    files = os.listdir(work_dir + job_id)
    work = []
    for file in files:
        work_file = open(work_dir + job_id + "/" + file, 'r').read()
        work.append(json.loads(work_file))
    return work


def get_available_jobs(): # get a list of jobs that are available to be worked on
    available_jobs = [job for job in job_cache.values() if get_job_being_worked_on(job["id"]) == False and len(get_job_work(job["id"])) < minimum_work]
    return available_jobs

@api.route('/api/batch', methods=['GET'])
def batch(): # return a batch of jobs to be worked on
    session = request.args.get('session')
    valid = verify_session(session)
    if not valid:
        return "invalid session"
    available_jobs = get_available_jobs()
    jobs = []
    if len(available_jobs) > 0:
        for i in range(batch_size):
            if i >= len(available_jobs):
                break
            job = random.randint(0, len(available_jobs)-1)
            work_being_done.append((available_jobs[job]["id"], session))
            jobs.append(available_jobs[job])
    print("Sending batch of " + str(len(jobs)) + " jobs from " + str(len(available_jobs)) + " available jobs")
    return jsonify(jobs)
# DOCS: GET /api/batch - returns [job, job, job...] - returns a batch of jobs to be worked on

def get_job_being_worked_on(job_id): # check if a job is being worked on
    for job in work_being_done:
        if job[0] == job_id:
            return True
    return False

@api.route('/api/job/<id>', methods=['POST'])
def job(id): # post the work done for a given job id
    job = job_cache[id]
    work = request.get_json()
    valid = verify_session(work["session"])
    if not valid:
        return "invalid session"
    if job == None:
        return "Job not found", 404
    work_id = len(os.listdir(work_dir + id + "/"))
    work_being_done.remove((id, work["session"]))
    work["id"] = work_id
    work["string"] = tokenizer.decode(work["tokens"])
    print("Received work for job " + id + " with id " + str(work_id))
    with open(work_dir + id + "/" + str(work_id+1) + ".json", 'w') as f:
        json.dump(work, f)
    return "ok"
# DOCS: POST /api/job/<id> - returns ok - body: {session: <session_id>, tokens: [tokens]} - tokens are the tokens generated by the worker

@api.route('/api/job/<id>', methods=['GET'])
def get_job(id): # return the total work done for a given job id
    job = job_cache[id]
    if job == None:
        return "Job not found", 404
    work = get_job_work(id)
    return jsonify({
        "id": id,
        "work": work,
        "tokenized": job["tokenized"]
    })
# DOCS: GET /api/job/<id> - returns {id: <id>, work: [work, work, work...], tokenized: [tokens]} - work is the work done by the workers, tokens are the tokens generated by the server

@api.route('/api/job', methods=['POST'])
def new_job(): # create a new job
    job = request.get_json()
    job_prompt = job["prompt"]
    if "id" in job:
        id = job["id"]
        if os.path.exists(job_dir + id + ".txt"):
            return "Job already exists", 400
    else:
        id = str(uuid.uuid4())
    with open(job_dir + id + ".txt", 'w') as f:
        f.write(job_prompt)
    work = {
        "id": id,
        "prompt": job_prompt,
        "tokenized": tokenizer.encode(job_prompt, add_special_tokens=False),
    }
    job_cache[id] = work
    os.makedirs(work_dir + id, exist_ok=True)
    return jsonify(work)
# DOCS: POST /api/job - returns {id: <id>, prompt: <prompt>, tokenized: [tokens]} - body: {prompt: <prompt>, id: <id>} - tokens are the tokens generated by the server - id is optional

@api.route('/api/session', methods=['GET'])
def session(): # create a new session
    id = str(uuid.uuid4())
    with open("./verified", 'a') as f:
        f.write(id + "\n")
    print("Created session " + id)
    return jsonify({
        "id": id
    })
# DOCS: GET /api/session - returns {id: <id>} - id is the session id

if __name__ == "__main__":
    get_job_cache() # load all jobs into memory
    print('http://localhost:3000')
    api.run(host='0.0.0.0', port=3000) # start the server