import logging
from typing import List
import torch
import transformers


def load_model(model, local_files_only):
    """ Load pretrained language model """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, local_files_only=local_files_only)
    config = transformers.AutoConfig.from_pretrained(model, output_hidden_states=True,
                                                     local_files_only=local_files_only)
    lm = transformers.AutoModelForMaskedLM.from_pretrained(model, config=config, local_files_only=local_files_only)
    lm.eval()
    parallel = torch.cuda.device_count() > 1
    if parallel:
        lm = torch.nn.DataParallel(lm)
    device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
    lm.to(device)
    logging.debug(f'running on {torch.cuda.device_count()} GPUs')
    return tokenizer, config, lm, device, parallel


class Dataset(torch.utils.data.Dataset):
    """ `torch.utils.data.Dataset` """
    float_tensors = ['attention_mask']

    def __init__(self, data: List):
        self.data = data  # a list of dictionaries

    def __len__(self):
        return len(self.data)

    def to_tensor(self, name, data):
        if name in self.float_tensors:
            return torch.tensor(data, dtype=torch.float32)
        return torch.tensor(data, dtype=torch.long)

    def __getitem__(self, idx):
        return {k: self.to_tensor(k, v) for k, v in self.data[idx].items()}
