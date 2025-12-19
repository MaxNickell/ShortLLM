"""REINFORCE-style RL trainer for ShortGPT."""

import os
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from src.config import RLConfig
from src.tokenizer import ShortGPTTokenizer
from src.rl.reward import compute_path_reward
from data.splits import get_stratified_subset
from .logger import log_metrics


class RLTrainer:
    """
    REINFORCE-style RL finetuning trainer.
    Uses policy gradient to optimize for shortest path generation.
    Includes early stopping based on validation avg_reward.
    """

    def __init__(
        self,
        model: nn.Module,
        config: RLConfig,
        tokenizer: ShortGPTTokenizer,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

    def _sample_path(self, row: dict) -> Tuple[str, torch.Tensor]:
        """Sample a path and return (generated_string, log_probs)."""
        prompt = (
            row["graph_repr"]
            + "<ORIGIN>" + str(row["origin"])
            + "<DEST>" + str(row["destination"])
            + "<START_PATH>"
        )

        input_ids = torch.tensor(
            [self.tokenizer.encode_string(prompt)],
            dtype=torch.long,
            device=self.device,
        )

        log_probs = []

        for _ in range(self.config.max_new_tokens):
            logits = self.model(input_ids)[:, -1, :]

            if self.config.temperature != 1.0:
                logits = logits / self.config.temperature

            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            next_id = dist.sample()
            log_probs.append(dist.log_prob(next_id))

            input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)

            if self.tokenizer.decode([next_id.item()])[0] == "<END_PATH>":
                break
            if input_ids.size(1) >= self.model.config.max_seq_len:
                break

        generated = "".join(self.tokenizer.decode(input_ids[0].tolist()))

        if not log_probs:
            return generated, torch.zeros(1, device=self.device, requires_grad=True)

        return generated, torch.stack(log_probs)

    def _generate_greedy(self, row: dict) -> str:
        """Generate path greedily (for evaluation)."""
        return self.model.generate(self.tokenizer, row, self.device, self.config.max_new_tokens)

    def _evaluate(self, val_subset: List[dict]) -> Tuple[float, float, float]:
        """
        Evaluate on validation subset.

        Returns:
            (avg_reward, optimal_rate, valid_suboptimal_rate)
        """
        self.model.eval()

        total_reward = 0.0
        optimal_count = 0
        valid_suboptimal_count = 0

        with torch.no_grad():
            for row in val_subset:
                generated = self._generate_greedy(row)
                reward = compute_path_reward(row, generated, self.tokenizer)

                total_reward += reward

                if reward == 2.0:
                    optimal_count += 1
                elif reward > 1.0:
                    valid_suboptimal_count += 1

        n = len(val_subset)
        avg_reward = total_reward / n
        optimal_rate = optimal_count / n
        valid_suboptimal_rate = valid_suboptimal_count / n

        return avg_reward, optimal_rate, valid_suboptimal_rate

    def train(self, train_rows: List[dict], val_rows: List[dict]):
        """Run RL finetuning with early stopping."""
        self.model.to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # Create stratified validation subset (fixed for all evals)
        val_subset = get_stratified_subset(val_rows, self.config.eval_samples, seed=42)
        print(f"Using stratified val subset of {len(val_subset)} samples")

        n = len(train_rows)
        global_step = 0
        best_val_reward = float("-inf")
        patience_counter = 0

        # Window stats for train metrics (reset every eval)
        window_rewards = []
        window_losses = []

        for epoch in range(1, self.config.num_epochs + 1):
            print(f"=== RL Epoch {epoch}/{self.config.num_epochs} ===")

            for step in range(1, self.config.steps_per_epoch + 1):
                self.model.train()
                global_step += 1

                # Sample batch
                batch_rows = [train_rows[random.randint(0, n - 1)] for _ in range(self.config.batch_size)]

                log_prob_sums = []
                rewards = []

                for row in batch_rows:
                    generated, log_probs = self._sample_path(row)
                    reward = compute_path_reward(row, generated, self.tokenizer)
                    rewards.append(reward)
                    log_prob_sums.append(log_probs.sum())

                rewards_t = torch.tensor(rewards, device=self.device)
                log_prob_sums_t = torch.stack(log_prob_sums)

                # REINFORCE with leave-one-out baseline (unbiased)
                if self.config.use_baseline and rewards_t.numel() > 1:
                    b = (rewards_t.sum() - rewards_t) / (rewards_t.numel() - 1)
                    advantages = rewards_t - b
                else:
                    advantages = rewards_t

                loss = -(advantages.detach() * log_prob_sums_t).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                optimizer.step()

                # Track window stats
                window_rewards.append(rewards_t.mean().item())
                window_losses.append(loss.item())

                # Print progress
                if global_step % self.config.log_every == 0:
                    avg_reward = rewards_t.mean().item()
                    print(f"Step {global_step}: avg_reward={avg_reward:.4f}, loss={loss.item():.4f}")

                # Evaluation and early stopping
                if global_step % self.config.eval_every == 0:
                    # Compute train metrics over window
                    train_avg_reward = sum(window_rewards) / len(window_rewards)
                    train_loss = sum(window_losses) / len(window_losses)

                    # Compute val metrics
                    val_avg_reward, val_optimal_rate, val_valid_suboptimal_rate = self._evaluate(val_subset)

                    print(f"Step {global_step} [EVAL]: "
                          f"train_reward={train_avg_reward:.4f}, "
                          f"val_reward={val_avg_reward:.4f}, "
                          f"val_optimal={val_optimal_rate:.2%}, "
                          f"val_suboptimal={val_valid_suboptimal_rate:.2%}")

                    # Log all metrics
                    log_metrics(
                        self.config.log_path,
                        step=global_step,
                        epoch=epoch,
                        train_avg_reward=train_avg_reward,
                        train_loss=train_loss,
                        val_avg_reward=val_avg_reward,
                        val_optimal_rate=val_optimal_rate,
                        val_valid_suboptimal_rate=val_valid_suboptimal_rate,
                    )

                    # Reset window stats
                    window_rewards = []
                    window_losses = []

                    # Early stopping check
                    if val_avg_reward > best_val_reward + self.config.min_delta:
                        best_val_reward = val_avg_reward
                        patience_counter = 0
                        self._save(self.config.save_path)
                    else:
                        patience_counter += 1
                        print(f"No improvement. Patience: {patience_counter}/{self.config.patience}")

                        if patience_counter >= self.config.patience:
                            print(f"Early stopping at step {global_step}")
                            return

                    self.model.train()

            print(f"Epoch {epoch} complete")

    def _save(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Saved to {path}")
