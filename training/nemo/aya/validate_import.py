from aya import *
from transformers import AutoModelForCausalLM
import torch

if __name__ == '__main__':
    nemo_model, _ = HFCohereExporter().nemo_load('/home/ubuntu/nvme/.cache/nemo/models/CohereLabs/aya-expanse-32b', cpu=False)
    nemo_model.eval()

    hf_model = AutoModelForCausalLM.from_pretrained('CohereLabs/aya-expanse-32b', device_map='cuda', torch_dtype=torch.float16)
    hf_model.eval()

    inputs = dict(input_ids=torch.arange(10, 20).unsqueeze(0).cuda(), position_ids=torch.arange(10).unsqueeze(0).cuda())

    nemo_output = nemo_model(**inputs)
    hf_output = hf_model(**inputs).logits

    print(nemo_output)
    print(hf_output)

    print('torch.allclose, tolerance = 1e-3:')
    print(torch.allclose(nemo_output, hf_output, atol=0.05))

    print(f"Mean abs diff: {torch.mean(torch.abs(nemo_output - hf_output))}")
    print(f"Max abs diff: {torch.max(torch.abs(nemo_output - hf_output))}")
