import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
    print(f'CUDA cache cleared for {torch.cuda.device_count()} devices')
else:
    print('CUDA not available')
