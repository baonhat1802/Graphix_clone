import argparse
import os
import math
import time
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.load_pt_dataset import PretrainDataset
from utils.load_sft_dataset import SFTSQLGenerationDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate.utils import set_seed
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from tqdm import tqdm


model = AutoModelForCausalLM.from_pretrained("bigcode/starcoderbase-1b",token="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",attn_implementation="flash_attention_2",torch_dtype=torch.float32)
model.to("cuda")

print(model)

# dataset = PretrainDataset("./tokenized_corpus.bin", 8192)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# # for batch in tqdm(dataloader):
# #     with open("log.txt", "a") as f:
# #         output = model(**batch)
# #         torch.save(output, "output.pt")
# #         break
# #         # f.write(str(batch))
# #     # break

# # print(output)


# output = torch.load("./output.pt")
# for i in output.items():
#     print(i, len(i))
