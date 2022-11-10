# RWKV Distributed Inference Server and Clients
Uses: https://github.com/BlinkDL/RWKV-LM

## What is this?

RWKV-LM is a Language Model that competes in output quality with GPT-Neo and GPT-J, with some key differences. One of them being how much more efficient it is to run on normal consumer hardware at decent speeds. This is insteaded to be a starting ground server that works out of the box for assigning prompts for completion to an arbitrary number of clients communicating with the server for work to inference completions for.

This uses the 3B RWKV Model as is, but 7B or 14B could easily be swapped in with minor changes to the client.py scripts. But the run requirements are much higher for GPUs(15GB VRAM required for 7B according to BlinkDL), hence why I went with 3B for this example. But larger models will still work on CPU just fine! They'll need more RAM, and run slower, but they're definitely usable at some level.

Think https://stablehorde.net/, but instead of inferencing Stable Diffusion, it's inferencing RWKV.

CPU's use 1 thread on average at 100% usage(on my 5950X at least) and tend to use 6GB of RAM last I checked, and averages 1 token/s with the current cuda script's fallback code (again on my 5950X, your milage may vary)

GPU uses a little less than 6GB of VRAM, averages 45 tokens/s.

## How To Use

Run the work server and assign work using the REST API(Documentation throughout code)

Run as many clients as desired, clients will collect batches of work from the work server and work on them, then return the results.

Each client has an associated README.

Very WIP/Proof of Concept/Take this and build what you really need off of it using this as a base kinda release.
