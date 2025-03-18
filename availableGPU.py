import torch

# Check if PyTorch is installed
print("PyTorch Version:", torch.__version__)

# Check if CUDA is available
print("CUDA Available:", torch.cuda.is_available())

# Check the installed CUDA version
print("CUDA Version:", torch.version.cuda)

# Check GPU device name
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())
else:
    print("CUDA is not available. PyTorch is running on CPU.")
