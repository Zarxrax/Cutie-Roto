@echo off
pushd %~dp0

echo Select version to install:
echo 1. CUDA (Fast processing if you have an NVIDIA GPU)
echo 2. CPU (Slow performance but smaller download)
echo 3. Exit

choice /C 123 /N /M "Enter your choice (1, 2, or 3):"

if errorlevel 3 (
    echo Exiting without installing dependencies
    exit /b 0
) else if errorlevel 2 (
    .\python-3.11.7-embed-amd64\python.exe -m pip install --upgrade pip --no-warn-script-location
    echo Uninstalling existing Pytorch
    .\python-3.11.7-embed-amd64\python.exe -m pip uninstall torch torchvision
    echo Installing CPU version of PyTorch
    .\python-3.11.7-embed-amd64\python.exe -m pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cpu
	.\python-3.11.7-embed-amd64\python.exe -m pip install -r .\requirements.txt
) else (
    .\python-3.11.7-embed-amd64\python.exe -m pip install --upgrade pip --no-warn-script-location
    echo Uninstalling existing Pytorch
    .\python-3.11.7-embed-amd64\python.exe -m pip uninstall torch torchvision
    echo Installing CUDA version of PyTorch
    .\python-3.11.7-embed-amd64\python.exe -m pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
	.\python-3.11.7-embed-amd64\python.exe -m pip install -r .\requirements.txt
)

echo Completed.
Pause