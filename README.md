# ShortGPT

A decoder-only transformer for shortest path prediction in graphs. This project investigates whether reinforcement learning (algorithmic alignment) improves performance over supervised pretraining.

## Task

Given a graph serialized as a string and an origin-destination pair, predict the shortest path.

**Input format:**
```
<EDGE>1<BD>2<EDGE>2<BD>3<EDGE>1<BD>3<ORIGIN>1<DEST>3<START_PATH>
```

**Output:**
```
1<TO>3<END_PATH>
```

## Training

### Phase 1: Supervised Pretraining

Train with cross-entropy loss on path tokens only (the graph representation is provided as context).

### Phase 2: RL Finetuning

Fine-tune using REINFORCE with a leave-one-out baseline. A dense reward function provides graduated feedback: invalid outputs receive negative rewards, valid but suboptimal paths receive moderate positive rewards, and optimal paths receive the highest rewards.

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Valid Structure** | Output follows `<START_PATH>node<TO>...<TO>node<END_PATH>` format |
| **Valid Path** | Valid structure AND correct origin/destination AND all edges exist in graph |
| **Optimal** | Valid path AND path length equals shortest path length |

Note: These metrics are hierarchical — valid path requires valid structure, and optimal requires valid path.

## Usage

```bash
# Pretrain
python train_pretrain.py --data data/processed/merged_final.jsonl

# RL finetune
python train_rl.py --data data/processed/merged_final.jsonl --checkpoint checkpoints/pretrained.pt

# Evaluate on test set (outputs to logs/eval_<checkpoint_name>.jsonl)
PYTHONPATH=. python scripts/eval_test.py --data data/processed/merged_final.jsonl --checkpoint checkpoints/pretrained.pt
PYTHONPATH=. python scripts/eval_test.py --data data/processed/merged_final.jsonl --checkpoint checkpoints/rl_finetuned.pt
```

## Configuration

Edit `src/config.py` to modify hyperparameters:

- `ShortGPTConfig`: model architecture (d_model, n_layers, n_heads, etc.)
- `DataConfig`: train/val/test split fractions and seed
- `PretrainConfig`: pretraining hyperparameters (lr, batch_size, patience, etc.)
- `RLConfig`: RL hyperparameters (lr, temperature, steps_per_epoch, etc.)

## Project Structure

```
├── train_pretrain.py      # Pretraining entry point
├── train_rl.py            # RL finetuning entry point
├── src/
│   ├── config.py          # All configurations
│   ├── tokenizer.py       # Fixed vocabulary tokenizer
│   ├── model/             # Transformer architecture
│   ├── training/          # Trainer classes
│   └── rl/reward.py       # Reward function
├── data/
│   ├── dataset.py         # PyTorch dataset
│   └── splits.py          # Consistent train/val/test splitting
└── scripts/               # Evaluation and plotting utilities
```
