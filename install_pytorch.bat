@echo off
echo Select PyTorch version to install:
echo 1. CPU (Select this if you are unsure which option to choose)
echo 2. CUDA (For NVIDIA GPUs)
echo 3. Exit

choice /C 123 /N /M "Enter your choice (1, 2, or 3):"

if errorlevel 3 (
    echo Exiting without installing PyTorch
    exit /b 0
) else if errorlevel 2 (
    echo Uninstalling existing Pytorch
    .\python-3.11.7-embed-amd64\python.exe -m pip uninstall torch torchvision
    echo Installing CUDA version of PyTorch
    .\python-3.11.7-embed-amd64\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else (
    echo Uninstalling existing Pytorch
    .\python-3.11.7-embed-amd64\python.exe -m pip uninstall torch torchvision
    echo Installing CPU version of PyTorch
    .\python-3.11.7-embed-amd64\python.exe -m pip install torch torchvision
)

echo Installation Completed.
Pause