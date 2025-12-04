import torch
from torch.utils.data import DataLoader

from src.config import ShortGPTConfig
from src.tokenizer import ShortGPTTokenizer
from data.dataset import GraphPathDataset, GraphPathCollator
from src.model.short_gpt import ShortGPT

def main():
    config = ShortGPTConfig()
    tokenizer = ShortGPTTokenizer(vocab_size=config.vocab_size)

    print("Loading datasets...")
    train_ds, val_ds, test_ds = GraphPathDataset.from_jsonl_splits(
        path="data/processed/merged_2_15.jsonl",  # adjust path if needed
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        train_frac=0.8,
        val_frac=0.1,
        seed=42,
    )
    print("Datasets loaded.")
    print("Train size:", len(train_ds), "Val size:", len(val_ds), "Test size:", len(test_ds))

    collator = GraphPathCollator(pad_token_id=tokenizer.pad_token_id)

    print("Building dataloaders...")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collator)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, collate_fn=collator)
    print("Dataloaders ready.")

    # Try one batch to make sure this is fine
    print("Grabbing one batch...")
    batch = next(iter(train_loader))
    print("Got batch. input_ids shape:", batch["input_ids"].shape)

    model = ShortGPT(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Starting pretraining...")
    model.fit_pretrain(
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        num_epochs=2,
        lr=3e-4,
    )
    print("Done training.")

if __name__ == "__main__":
    main()
