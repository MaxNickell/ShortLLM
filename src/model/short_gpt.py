import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict

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
        x = self.token_emb(input_ids) # (B, T, d_model)

        # 2. Apply dropout
        x = self.drop(x)

        # 3. Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # 4. Final layernorm
        x = self.ln_f(x)

        # 5. Project to logits
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits
    
    def _path_only_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute cross-entropy loss only on tokens that belong to the path segment
        (<START_PATH> ... <END_PATH>), ignoring all other tokens and any padding.

        batch is expected to have:
          - "input_ids"       : (B, T) LongTensor
          - "path_token_mask" : (B, T) BoolTensor
        """
        input_ids = batch["input_ids"]
        path_mask = batch["path_token_mask"]

        # Forward pass: logits (B, T, vocab_size)
        logits = self(input_ids)

        # Shift for next-token prediction:
        # we predict token at position t+1 from positions <= t
        logits = logits[:, :-1, :]
        labels = input_ids[:, 1:]
        path_mask = path_mask[:, 1:]

        B, T_minus1, V = logits.size()

        # Flatten for CrossEntropyLoss
        logits_flat = logits.reshape(B * T_minus1, V)
        labels_flat = labels.reshape(B * T_minus1)
        mask_flat = path_mask.reshape(B * T_minus1)

        # Per-token loss
        criterion = nn.CrossEntropyLoss(reduction="none")
        per_token_loss = criterion(logits_flat, labels_flat)

        # Mask out non-path tokens
        masked_loss = per_token_loss * mask_flat.float()

        # Normalize by number of path tokens
        num_path_tokens = mask_flat.float().sum()
        if num_path_tokens.item() == 0:
            # Should not happen with your data, but guard anyway
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = masked_loss.sum() / num_path_tokens
        return loss
    
    def fit_pretrain(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: ShortGPTTokenizer,
        device: torch.device,
        num_epochs: int = 5,
        lr: float = 3e-4,
    ) -> "ShortGPT":
        """
        Supervised pretraining for Model 1:

          - Optimizes path-only cross-entropy on the train set.
          - Evaluates path-only loss on the val set each epoch.
          - Returns self so you can chain or save.
        """
        self.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        for epoch in range(1, num_epochs + 1):
            # ---- Train ----
            self.train()
            total_train_loss = 0.0
            num_train_batches = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                path_mask = batch["path_token_mask"].to(device)
                batch_on_device = {
                    "input_ids": input_ids,
                    "path_token_mask": path_mask,
                }

                optimizer.zero_grad()
                loss = self._path_only_loss(batch_on_device)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                num_train_batches += 1

            avg_train_loss = total_train_loss / max(1, num_train_batches)

            # ---- Validation ----
            self.eval()
            total_val_loss = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    path_mask = batch["path_token_mask"].to(device)
                    batch_on_device = {
                        "input_ids": input_ids,
                        "path_token_mask": path_mask,
                    }

                    loss = self._path_only_loss(batch_on_device)
                    total_val_loss += loss.item()
                    num_val_batches += 1

            avg_val_loss = total_val_loss / max(1, num_val_batches)

            print(
                f"Epoch {epoch}/{num_epochs} "
                f"- train loss: {avg_train_loss:.4f} "
                f"- val loss: {avg_val_loss:.4f}"
            )

        return self

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

