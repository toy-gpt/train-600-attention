"""c_model.py - Simple model module.

Defines a minimal next-token prediction model for a context-3 setting
(uses three tokens in sequence as context).

Responsibilities:
- Represent a simple parameterized model that maps a
  3-tuple of token IDs (prev2, prev1, current)
  to a score for each token in the vocabulary.
- Convert scores into probabilities using softmax.
- Provide a forward pass (no training in this module).

This model is intentionally simple:
- one weight table (conceptually a 4D tensor: prev2 x prev1 x curr x next,
  flattened for storage)
- one forward computation
- no learning here

Training is handled in a different module.
"""

import logging
import math
from typing import Final

from datafun_toolkit.logger import get_logger, log_header

LOG: logging.Logger = get_logger("MODEL", level="INFO")


class SimpleNextTokenModel:
    """A minimal next-token prediction model (context-3)."""

    def __init__(self, vocab_size: int) -> None:
        """Initialize the model with random weights."""
        self.vocab_size: Final[int] = vocab_size

        # Conceptual shape:
        #   vocab_size x vocab_size x vocab_size x vocab_size
        #   (prev2, prev1, current) -> next
        #
        # Stored as:
        #   (vocab_size ** 3) rows x vocab_size columns
        self.weights: list[list[float]] = [
            [0.0 for _ in range(vocab_size)] for _ in range(vocab_size**3)
        ]

        LOG.info(f"Model initialized with vocabulary size {vocab_size} (context-3).")

    def _row_index(self, prev2_id: int, prev1_id: int, current_id: int) -> int:
        return (
            prev2_id * self.vocab_size * self.vocab_size
            + prev1_id * self.vocab_size
            + current_id
        )

    def forward(
        self,
        prev2_id: int,
        prev1_id: int,
        current_id: int,
    ) -> list[float]:
        """Perform a forward pass to get next-token probabilities.

        Args:
            prev2_id: Token ID of the token two positions before current.
            prev1_id: Token ID of the token one position before current.
            current_id: Token ID of the current token.

        Returns:
            list[float]: Probabilities for each token in the vocabulary.
        """
        row_idx = self._row_index(prev2_id, prev1_id, current_id)
        scores = self.weights[row_idx]
        return self._softmax(scores)

    @staticmethod
    def _softmax(scores: list[float]) -> list[float]:
        """Convert raw scores into probabilities.

        Args:
            scores: Raw score values.

        Returns:
            Normalized probability distribution.
        """
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        return [s / total for s in exp_scores]


def main() -> None:
    """Demonstrate a forward pass of the simple context-3 model."""
    # Local imports keep modules decoupled.
    from toy_gpt_train.a_tokenizer import SimpleTokenizer
    from toy_gpt_train.b_vocab import Vocabulary

    log_header(LOG, "Simple Next-Token Model Demo (Context-3)")

    # Step 1: Tokenize input text.
    tokenizer: SimpleTokenizer = SimpleTokenizer()
    tokens: list[str] = tokenizer.get_tokens()

    if len(tokens) < 3:
        LOG.info("Need at least three tokens for context-3 demonstration.")
        return

    # Step 2: Build vocabulary.
    vocab: Vocabulary = Vocabulary(tokens)

    # Step 3: Initialize model.
    model: SimpleNextTokenModel = SimpleNextTokenModel(vocab_size=vocab.vocab_size())

    # Step 4: Select context tokens (prev2, prev1, current).
    prev2_token: str = tokens[0]
    prev1_token: str = tokens[1]
    current_token: str = tokens[2]

    prev2_id: int | None = vocab.get_token_id(prev2_token)
    prev1_id: int | None = vocab.get_token_id(prev1_token)
    current_id: int | None = vocab.get_token_id(current_token)

    if prev2_id is None or prev1_id is None or current_id is None:
        LOG.info("One of the sample tokens was not found in vocabulary.")
        return

    # Step 5: Forward pass (context-3).
    probs: list[float] = model.forward(prev2_id, prev1_id, current_id)

    # Step 6: Inspect results.
    LOG.info(
        f"Input tokens: {prev2_token!r} (ID {prev2_id}), {prev1_token!r} (ID {prev1_id}), {current_token!r} (ID {current_id})"
    )
    LOG.info("Output probabilities for next token:")
    for idx, prob in enumerate(probs):
        tok: str | None = vocab.get_id_token(idx)
        LOG.info(f"  {tok!r} (ID {idx}) -> {prob:.4f}")


if __name__ == "__main__":
    main()
