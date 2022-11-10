# RWKV Distributed Inference Server and Clients
Uses: https://github.com/BlinkDL/RWKV-LM

## What is this?

RWKV-LM is a Language Model that competes in output quality with GPT, with some key differences. One of them being how much more efficient it is to run on normal consumer hardware at decent speeds. This is insteaded to be a starting ground server that works out of the box for assigning prompts for completion to an arbitrary number of clients communicating with the server for work to inference completions for.

Think https://stablehorde.net/, but instead of inferencing Stable Diffusion, it's inferencing RWKV.

CPU's use 1 thread on average at 100% usage(on my 5950X at least) and tend to use 6GB of RAM last I checked, and averages 1 token/s with the current cuda script's fallback code (again on my 5950X, your milage may vary)

GPU uses a little less than 6GB of VRAM, averages 45 tokens/s.

## How To Use

Run the work server and assign work using the REST API(Documentation throughout code)

Run as many clients as desired, clients will collect batches of work from the work server and work on them, then return the results.

Each client has an associated README.

Very WIP/Proof of Concept/Take this and build what you really need off of it using this as a base kinda release.
