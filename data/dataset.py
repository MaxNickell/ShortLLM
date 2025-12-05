import json
import random
from math import floor
from typing import List, Dict, Any, Tuple
import torch
from torch.utils.data import Dataset

from src.tokenizer import ShortGPTTokenizer


class GraphPathDataset(Dataset):
    """
    PyTorch Dataset for shortest-path pretraining.

    Responsibilities:
      - Hold raw JSON rows for this task.
      - Build the combined sequence string for each row:
      - Use ShortGPTTokenizer to tokenize it.
      - Output:
          input_ids       : LongTensor [T]
          path_token_mask : BoolTensor [T] (True where token is in the path segment)
          length          : LongTensor scalar (T)
    """

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tokenizer: ShortGPTTokenizer,
        max_seq_len: int,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]

        # Build full string:
        graph_repr = row["graph_repr"]
        origin = row["origin"]
        dest = row["destination"]
        serialized_path = row["serialized_path"]

        full_str = (
            graph_repr
            + "<ORIGIN>" + str(origin)
            + "<DEST>" + str(dest)
            + serialized_path
        )

        # Tokenize to token list, then to ids
        tokens = self.tokenizer.tokenize_string(full_str)
        token_ids = self.tokenizer.encode(tokens)

        if len(token_ids) > self.max_seq_len:
            raise ValueError(
                f"Sequence length {len(token_ids)} exceeds max_seq_len {self.max_seq_len}"
            )

        # Build mask for path tokens:
        # from <START_PATH> through <END_PATH> inclusive
        path_token_mask = [False] * len(tokens)

        try:
            start_idx = tokens.index("<START_PATH>")
        except ValueError:
            raise ValueError("No <START_PATH> token found in sequence.")

        try:
            end_idx = tokens.index("<END_PATH>")
        except ValueError:
            raise ValueError("No <END_PATH> token found in sequence.")

        if end_idx < start_idx:
            raise ValueError("Found <END_PATH> before <START_PATH>.")

        for i in range(start_idx, end_idx + 1):
            path_token_mask[i] = True

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        path_token_mask = torch.tensor(path_token_mask, dtype=torch.bool)

        return {
            "input_ids": input_ids,
            "path_token_mask": path_token_mask,
            "length": torch.tensor(len(token_ids), dtype=torch.long),
        }

    @classmethod
    def _load_rows_from_jsonl(cls, path: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    @classmethod
    def _split_indices(
        cls,
        n: int,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[int], List[int], List[int]]:
        rng = random.Random(seed)
        idxs = list(range(n))
        rng.shuffle(idxs)

        n_train = floor(train_frac * n)
        n_val = floor(val_frac * n)
        n_test = n - n_train - n_val

        train_idxs = idxs[:n_train]
        val_idxs = idxs[n_train:n_train + n_val]
        test_idxs = idxs[n_train + n_val:]

        return train_idxs, val_idxs, test_idxs

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        tokenizer: ShortGPTTokenizer,
        max_seq_len: int,
    ) -> "GraphPathDataset":
        """Load ALL rows from a JSONL file into a single Dataset."""
        rows = cls._load_rows_from_jsonl(path)
        return cls(rows=rows, tokenizer=tokenizer, max_seq_len=max_seq_len)

    @classmethod
    def from_jsonl_splits(
        cls,
        path: str,
        tokenizer: ShortGPTTokenizer,
        max_seq_len: int,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        seed: int = 42,
    ) -> Tuple["GraphPathDataset", "GraphPathDataset", "GraphPathDataset"]:
        """
        Load rows from JSONL and return (train_ds, val_ds, test_ds)
        using an 80/10/10-style split by examples.
        """
        rows = cls._load_rows_from_jsonl(path)
        n = len(rows)
        train_idxs, val_idxs, test_idxs = cls._split_indices(
            n=n,
            train_frac=train_frac,
            val_frac=val_frac,
            seed=seed,
        )

        def subset(indices: List[int]) -> List[Dict[str, Any]]:
            return [rows[i] for i in indices]

        train_rows = subset(train_idxs)
        val_rows = subset(val_idxs)
        test_rows = subset(test_idxs)

        train_ds = cls(train_rows, tokenizer=tokenizer, max_seq_len=max_seq_len)
        val_ds = cls(val_rows, tokenizer=tokenizer, max_seq_len=max_seq_len)
        test_ds = cls(test_rows, tokenizer=tokenizer, max_seq_len=max_seq_len)

        return train_ds, val_ds, test_ds


class GraphPathCollator:
    """
    Collate function for GraphPathDataset.

    - Pads sequences in a batch to the same length (right-padding).
    - Uses tokenizer.pad_token_id as the padding value.
    - We will later mask out padded positions in the loss.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        lengths = [item["length"].item() for item in batch]
        max_len = max(lengths)

        input_ids = torch.full(
            (batch_size, max_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        path_token_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.bool,
        )

        for i, item in enumerate(batch):
            L = item["length"].item()
            input_ids[i, :L] = item["input_ids"]
            path_token_mask[i, :L] = item["path_token_mask"]

        return {
            "input_ids": input_ids,
            "path_token_mask": path_token_mask,
            "lengths": torch.tensor(lengths, dtype=torch.long),
        }
