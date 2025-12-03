# ============================ Define Training and Evaluation Functions ============================= #
# functions to train and evaluate the BERT model, handling forward and backward passes, loss computation, and metric calculations.
# Which abstracts the training and evaluation steps to keep the main script clean

from sched import scheduler
import torch.nn as nn
import torch
from tqdm import tqdm



# No module-level loss function; pass as parameter to functions

bce_loss = nn.BCEWithLogitsLoss()

def train_fn(data_loader, model, optimizer, device, accumulation_steps, scheduler, loss_fn):
    # set model to training mode (enables dropout)
    model.train()
    # clear gradients
    optimizer.zero_grad()
    
    total_batches = len(data_loader)
    print(f"[train_fn] Starting training with {total_batches} batches")
    
    batch_count = 0
    for bi, d in enumerate(data_loader):
        batch_count += 1
        if batch_count == 1:
            print(f"[train_fn] First batch received. Batch type: {type(d)}, keys: {d.keys()}")
        
        try:
            ids = d['ids'].to(device, dtype=torch.long) # move tensors to GPU/CPU and convert to long (integer) dtype
            token_type_ids = d['token_type_ids'].to(device, dtype=torch.long)
            mask = d['mask'].to(device, dtype=torch.long)
            targets = d['targets'].to(device, dtype=torch.float)

            if bi < 5:
                print(f"[train_fn] Batch {bi+1}: calling model forward on device={device}")

            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                token_type_ids=token_type_ids
            )

            if bi < 5:
                try:
                    print(f"[train_fn] Batch {bi+1}: model output shape: {outputs.shape}")
                except Exception:
                    print(f"[train_fn] Batch {bi+1}: could not read output shape (type={type(outputs)})")

            # compute loss
            loss = loss_fn(outputs, targets)
            if bi < 5:
                print(f"[train_fn] Batch {bi+1}: loss computed: {loss.item()}")
            # backward pass
            loss.backward()
            if bi < 5:
                print(f"[train_fn] Batch {bi+1}: backward complete")

            # Temporarily force the optimizer step on the first batch to debug
            # if (bi + 1) == 1:
            #     print(f"[train_fn] Batch {bi+1}: Forcing optimizer step for debug.")
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            #     optimizer.step()
            #     scheduler.step()
            #     optimizer.zero_grad()
            #     break # Stop after the first step for quick diagnosis

            # run forward pass
            if (bi + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if (bi + 1) % (accumulation_steps * 100) == 0:
                    print(f"  [Batch {bi + 1}/{total_batches}] Loss: {loss.item():.4f}")
        except Exception as e:
            print(f"\n[ERROR] Exception at batch {bi}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise

    # Handle last batch if not a multiple of accumulation_steps
    if len(data_loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    print(f"[train_fn] Epoch training complete.")


def eval_fn(data_loader, model, device, loss_fn=bce_loss):
    model.eval()
    fin_outputs = []
    fin_targets = []
    with torch.no_grad():
        for d in tqdm(data_loader, total=len(data_loader)):
            ids = d['ids'].to(device, dtype=torch.long)
            token_type_ids = d['token_type_ids'].to(device, dtype=torch.long)
            mask = d['mask'].to(device, dtype=torch.long)
            targets = d['targets'].to(device, dtype=torch.float)

            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                token_type_ids=token_type_ids
            )

            # Use in-place sigmoid and move to CPU in batches
            batch_outputs = torch.sigmoid(outputs).cpu().detach().numpy()
            batch_targets = targets.cpu().detach().numpy()
            fin_outputs.extend(batch_outputs.tolist())
            fin_targets.extend(batch_targets.tolist())
    
    return fin_outputs, fin_targets
