@echo off
echo Select version to install:
echo 1. CPU (Select this if you are unsure which option to choose)
echo 2. CUDA (Faster, only for NVIDIA GPUs)
echo 3. Exit

choice /C 123 /N /M "Enter your choice (1, 2, or 3):"

if errorlevel 3 (
    echo Exiting without installing dependencies
    exit /b 0
) else if errorlevel 2 (
    echo Uninstalling existing Pytorch
    .\python-3.11.7-embed-amd64\python.exe -m pip uninstall torch torchvision
    echo Installing CUDA version of PyTorch
    .\python-3.11.7-embed-amd64\python.exe -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
	.\python-3.11.7-embed-amd64\python.exe -m pip install -r .\requirements.txt
) else (
    echo Uninstalling existing Pytorch
    .\python-3.11.7-embed-amd64\python.exe -m pip uninstall torch torchvision
    echo Installing CPU version of PyTorch
    .\python-3.11.7-embed-amd64\python.exe -m pip install torch==2.3.1 torchvision==0.18.1
	.\python-3.11.7-embed-amd64\python.exe -m pip install -r .\requirements.txt
)

echo Completed.
Pause