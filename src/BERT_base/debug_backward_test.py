# Quick diagnostic: run one forward + backward on CPU and report errors/memory
import sys
import os
# Limit OpenMP threads to avoid threading-related crashes
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import torch
torch.set_num_threads(1)

sys.path.insert(0, 'src')
import config
import pandas as pd
from dataset import BERTDataset
from torch.utils.data import DataLoader
from model import BERTBaseUncased
import traceback

# optional psutil for memory info
try:
    import psutil
    have_psutil = True
except Exception:
    have_psutil = False

if have_psutil:
    vm = psutil.virtual_memory()
    print(f"Memory before: total={vm.total}, available={vm.available}")
else:
    print("psutil not available; skipping memory info")

# Prepare tiny dataset
df = pd.read_csv(config.TRAINING_FILE).fillna('none')
# ensure sentiment converted same as train
if 'sentiment' in df.columns and df.sentiment.dtype == 'object':
    df.sentiment = df.sentiment.apply(lambda x: 1 if x == 'positive' else 0)

reviews = df.review.values[:4]
targets = df.sentiment.values[:4]

ds = BERTDataset(reviews, targets)
loader = DataLoader(ds, batch_size=4, num_workers=0)
batch = next(iter(loader))
print('Loaded batch shapes:', batch['ids'].shape, batch['mask'].shape, batch['targets'].shape)

# Run model forward+backward on CPU
device = torch.device('cpu')
model = BERTBaseUncased().to(device)
model.train()

try:
    print('Calling forward...')
    out = model(input_ids=batch['ids'].to(device), attention_mask=batch['mask'].to(device), token_type_ids=batch['token_type_ids'].to(device))
    print('Forward OK. Output shape:', out.shape)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = loss_fn(out, batch['targets'].to(device))
    print('Loss computed:', loss.item())
    print('Calling backward...')
    loss.backward()
    print('Backward OK (completed)')
except Exception as e:
    print('Exception during forward/backward:')
    traceback.print_exc()
    raise

if have_psutil:
    vm = psutil.virtual_memory()
    print(f"Memory after: total={vm.total}, available={vm.available}")

print('Diagnostic script finished normally')