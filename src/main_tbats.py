import torch
import os

import config
from experiment import forecast

if __name__ == "__main__":
    # Check the number of GPUs available
    num_gpus = torch.cuda.device_count()

    # Get the name of each GPU
    devices = [torch.cuda.get_device_name(i) for i in range(num_gpus)]

    # Print the devices
    for i, device in enumerate(devices):
        print(f"Device {i}: {device}")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only the first GPU.

    forecast.forecast_model(model_dir=config.scaled_model_dir, model_name="TBATS")
