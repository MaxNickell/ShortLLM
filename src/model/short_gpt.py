import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional

from src.config import ShortGPTConfig
from .block import TransformerBlock
from src.tokenizer import ShortGPTTokenizer


class ShortGPT(nn.Module):
    """
    Top-level GPT model for shortest-path reasoning.
    """

    def __init__(self, config: ShortGPTConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_emb = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
        )

        # Dropout
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Linear layer to project hidden states to vocabulary size
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )

        # Weight tying: token_emb and lm_head share parameters
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids):
        """
        input_ids: LongTensor of shape (B, T)
        Returns logits of shape (B, T, vocab_size)
        """
        B, T = input_ids.shape

        # 1. Embed tokens
        x = self.token_emb(input_ids)  # (B, T, d_model)

        # 2. Apply dropout
        x = self.drop(x)

        # 3. Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # 4. Final layernorm
        x = self.ln_f(x)

        # 5. Project to logits
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    def _path_loss_and_counts(self, batch: Dict[str, torch.Tensor]):
        """
        Compute:
          - mean cross-entropy loss over path tokens
          - number of correct path-token predictions
          - number of path tokens

        batch:
          - "input_ids"       : (B, T) LongTensor
          - "path_token_mask" : (B, T) BoolTensor
        """
        input_ids = batch["input_ids"]       # (B, T)
        path_mask = batch["path_token_mask"] # (B, T)

        # Forward pass: logits (B, T, vocab_size)
        logits = self(input_ids)

        # Shift for next-token prediction:
        # we predict token at position t+1 from positions <= t
        logits = logits[:, :-1, :]   # (B, T-1, V)
        labels = input_ids[:, 1:]    # (B, T-1)
        path_mask = path_mask[:, 1:] # (B, T-1)

        B, T_minus1, V = logits.size()

        # Flatten
        logits_flat = logits.reshape(B * T_minus1, V)      # (B*T-1, V)
        labels_flat = labels.reshape(B * T_minus1)         # (B*T-1,)
        mask_flat = path_mask.reshape(B * T_minus1)        # (B*T-1,)

        # Per-token CE loss
        criterion = nn.CrossEntropyLoss(reduction="none")
        per_token_loss = criterion(logits_flat, labels_flat)  # (B*T-1,)

        # Mask to path tokens only
        mask_float = mask_flat.float()
        masked_loss = per_token_loss * mask_float           # (B*T-1,)

        num_path_tokens = mask_float.sum()

        if num_path_tokens.item() == 0:
            # Should not happen with your data, but guard anyway
            zero = torch.tensor(0.0, device=logits.device)
            return zero, zero, zero

        # Mean loss over path tokens
        loss = masked_loss.sum() / num_path_tokens

        # Accuracy: compare argmax with labels, on path tokens only
        preds_flat = torch.argmax(logits_flat, dim=-1)       # (B*T-1,)
        correct_vec = ((preds_flat == labels_flat) & mask_flat).float()
        num_correct = correct_vec.sum()

        return loss, num_correct, num_path_tokens

    def fit_pretrain(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: ShortGPTTokenizer,  # kept for future use if needed
        device: torch.device,
        num_epochs: int = 20,
        lr: float = 3e-4,
        patience: int = 3,
        save_path: Optional[str] = "model1_pretrained.pt",
    ) -> dict:
        """
        Supervised pretraining for Model 1.

        - Optimizes path-only cross-entropy on the train set.
        - Evaluates path-only loss and path-token accuracy on the val set.
        - Uses early stopping based on val loss with `patience`.
        - Optionally saves the best checkpoint to `save_path`.
        - Returns a history dict with per-epoch train/val loss and accuracy.
        """
        self.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, num_epochs + 1):
            # ---- Training ----
            self.train()
            total_train_loss_weighted = 0.0
            total_train_tokens = 0.0
            total_train_correct = 0.0

            for batch_idx, batch in enumerate(train_loader, start=1):
                input_ids = batch["input_ids"].to(device)
                path_mask = batch["path_token_mask"].to(device)
                batch_on_device = {
                    "input_ids": input_ids,
                    "path_token_mask": path_mask,
                }

                optimizer.zero_grad()
                loss, num_correct, num_tokens = self._path_loss_and_counts(batch_on_device)
                loss.backward()
                optimizer.step()

                # Accumulate weighted loss and counts for global metrics
                total_train_loss_weighted += loss.item() * num_tokens.item()
                total_train_tokens += num_tokens.item()
                total_train_correct += num_correct.item()

                # Optional intra-epoch logging
                if batch_idx % 2000 == 0:
                    avg_so_far = total_train_loss_weighted / max(1.0, total_train_tokens)
                    acc_so_far = total_train_correct / max(1.0, total_train_tokens)
                    print(
                        f"Epoch {epoch} - batch {batch_idx} "
                        f"- avg train loss: {avg_so_far:.4f} "
                        f"- avg train acc: {acc_so_far*100:.2f}%"
                    )

            avg_train_loss = total_train_loss_weighted / max(1.0, total_train_tokens)
            train_acc = total_train_correct / max(1.0, total_train_tokens)

            # ---- Validation ----
            self.eval()
            total_val_loss_weighted = 0.0
            total_val_tokens = 0.0
            total_val_correct = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    path_mask = batch["path_token_mask"].to(device)
                    batch_on_device = {
                        "input_ids": input_ids,
                        "path_token_mask": path_mask,
                    }

                    loss, num_correct, num_tokens = self._path_loss_and_counts(batch_on_device)

                    total_val_loss_weighted += loss.item() * num_tokens.item()
                    total_val_tokens += num_tokens.item()
                    total_val_correct += num_correct.item()

            avg_val_loss = total_val_loss_weighted / max(1.0, total_val_tokens)
            val_acc = total_val_correct / max(1.0, total_val_tokens)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch}/{num_epochs} "
                f"- train loss: {avg_train_loss:.4f}, train acc: {train_acc*100:.2f}% "
                f"- val loss: {avg_val_loss:.4f}, val acc: {val_acc*100:.2f}%"
            )

            # ---- Early stopping & checkpointing ----
            if avg_val_loss < best_val_loss - 1e-4:  # small min_delta to avoid noise
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0

                if save_path is not None:
                    torch.save(self.state_dict(), save_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(
                        f"Early stopping triggered after {epoch} epochs "
                        f"(no val improvement for {patience} epochs)."
                    )
                    break

        return history


    def generate_path(
        self,
        tokenizer: ShortGPTTokenizer,
        row: dict,
        device: torch.device,
        max_new_tokens: int = 64,
    ) -> str:
        """
        Run autoregressive inference on a single (graph, origin, dest) row.

        We:
          - Build the prompt: graph_repr + <ORIGIN>o<DEST>d + <START_PATH>
            (we stop before the ground-truth path).
          - Autoregressively generate tokens until <END_PATH> or max_new_tokens.
          - Return the generated sequence as a string.
        """
        self.eval()
        self.to(device)

        graph_repr = row["graph_repr"]
        origin = row["origin"]
        dest = row["destination"]

        # Prompt: graph_repr + <ORIGIN>o<DEST>d + <START_PATH>
        prompt_str = (
            graph_repr
            + "<ORIGIN>" + str(origin)
            + "<DEST>" + str(dest)
            + "<START_PATH>"
        )

        # Tokenize prompt
        input_ids_list = tokenizer.encode_string(prompt_str)
        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)  # (1, T)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self(input_ids)
                next_logits = logits[:, -1, :]
                next_id = torch.argmax(next_logits, dim=-1)

                # Append next token
                input_ids = torch.cat(
                    [input_ids, next_id.unsqueeze(0)], dim=1
                )  # (1, T+1)

                # Check if we hit <END_PATH>
                next_token_str = tokenizer.decode([next_id.item()])[0]
                if next_token_str == "<END_PATH>":
                    break

                # Optional safety: don't exceed model's context window
                if input_ids.size(1) >= self.config.max_seq_len:
                    break

        # Decode full sequence back to tokens
        full_token_ids = input_ids[0].tolist()
        full_tokens = tokenizer.decode(full_token_ids)

        # Join back to string (you can customize formatting)
        generated_str = "".join(full_tokens)
        return generated_str

