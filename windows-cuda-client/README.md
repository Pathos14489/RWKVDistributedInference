The Windows version of the client: admittedly haven't really tested if this works yet? I might need to fix the run script, batch has never really been an interest of mine before so I mostly winged it and let copilot do the talking.

---

Download: https://huggingface.co/BlinkDL/rwkv-4-pile-3b/resolve/main/RWKV-4-Pile-3B-20221008-8023.pth

Place inside this folder with the run.bat file.

Has light CPU support in a pinch, but it's slow as balls currently. ONNX version should be 2X faster for CPU and should work on AMD GPUs.

Please use run.bat, don't run the client.py unless you've run run.bat at least once, or I guess you can if you know what you're doing or something, smarty pants.

Stay Tuned