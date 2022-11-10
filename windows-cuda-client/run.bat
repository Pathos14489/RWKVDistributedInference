:: Runs captioning script using included Python 3.10.0 under ./Python-win/Python/
:: Requires ./Python-win/Python/ to be extracted from Python 3.10.0 Windows x86-64 executable installer
@echo off
setlocal
:: Python-win/Python/ is extracted from Python 3.10.0 Windows x86-64 executable installer
set "PATH=%~dp0Python-win\Python;%PATH%"
pip install torch torchdynamo numpy tqdm
python client.py
endlocal