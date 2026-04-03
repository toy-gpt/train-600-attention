"""d_train.py - Training loop module.

Trains the SimpleNextTokenModel on a small token corpus
using a context-3 window (three preceding tokens).

Responsibilities:
- Create ((token_{t-2}, token_{t-1}, token_t) -> next_token) training pairs
- Run a basic gradient-descent training loop
- Track loss and accuracy per epoch
- Write a CSV log of training progress
- Write inspectable training artifacts (vocabulary, weights, embeddings, meta)

Concepts:
- softmax: converts raw scores into probabilities (so predictions sum to 1)
- cross-entropy loss: measures how well predicted probabilities match the correct token
- gradient descent: iterative weight updates to minimize loss
  - think descending to find the bottom of a valley in a landscape
  - where the valley floor corresponds to lower prediction error

Notes:
- This remains intentionally simple: no deep learning framework, no Transformer.
- The model generalizes n-gram training by expanding the context window.
- Training updates weight rows associated with the observed context-3 pattern.
- token_embeddings.csv remains a derived visualization artifact;
  learned embeddings are introduced in later stages.
"""

import logging
from pathlib import Path
from typing import Final

from datafun_toolkit.logger import get_logger, log_header, log_path

from toy_gpt_train.c_model import SimpleNextTokenModel
from toy_gpt_train.io_artifacts import (
    RowLabeler,
    VocabularyLike,
    find_single_corpus_file,
    write_artifacts,
    write_training_log,
)
from toy_gpt_train.math_training import argmax, cross_entropy_loss

LOG: logging.Logger = get_logger("TRAIN", level="INFO")

BASE_DIR: Final[Path] = Path(__file__).resolve().parents[2]
OUTPUTS_DIR: Final[Path] = BASE_DIR / "outputs"
TRAIN_LOG_PATH: Final[Path] = OUTPUTS_DIR / "train_log.csv"

type Context3 = tuple[int, int, int]
type Context3Pair = tuple[Context3, int]


def token_row_index_context3(context_ids: Context3, vocab_size: int) -> int:
    """Return the row index for a context-3 token sequence.

    Context order:
        (token_id_{t-2}, token_id_{t-1}, token_id_t)

    Flattening scheme:
        row_index = a * vocab_size^2 + b * vocab_size + c

    This is the context-3 analogue of:
        unigram: row = token_id
        bigram:  row = prev_id * vocab_size + curr_id
    """
    token_id_t_minus_2, token_id_t_minus_1, token_id_t = context_ids
    return (
        token_id_t_minus_2 * vocab_size * vocab_size
        + token_id_t_minus_1 * vocab_size
        + token_id_t
    )


def row_labeler_context3(vocab: VocabularyLike, vocab_size: int) -> RowLabeler:
    """Map a context-3 row index to a label like 'tok_{t-2}|tok_{t-1}|tok_t'."""

    def label(row_idx: int) -> str:
        token_id_t_minus_2: int = row_idx // (vocab_size * vocab_size)
        remainder: int = row_idx % (vocab_size * vocab_size)

        token_id_t_minus_1: int = remainder // vocab_size
        token_id_t: int = remainder % vocab_size

        tok2: str = vocab.get_id_token(token_id_t_minus_2) or f"id_{token_id_t_minus_2}"
        tok1: str = vocab.get_id_token(token_id_t_minus_1) or f"id_{token_id_t_minus_1}"
        tok0: str = vocab.get_id_token(token_id_t) or f"id_{token_id_t}"

        return f"{tok2}|{tok1}|{tok0}"

    return label


def make_training_pairs(token_ids: list[int]) -> list[Context3Pair]:
    """Convert token IDs into ((t-2, t-1, t), next) training pairs.

    Example:
        ids = [3, 1, 2, 4, 5]
        pairs = [((3, 1, 2), 4), ((1, 2, 4), 5)]
    """
    pairs: list[Context3Pair] = []
    for i in range(len(token_ids) - 3):
        context_ids: Context3 = (token_ids[i], token_ids[i + 1], token_ids[i + 2])
        next_id: int = token_ids[i + 3]
        pairs.append((context_ids, next_id))
    return pairs


def train_model(
    model: "SimpleNextTokenModel",
    pairs: list[Context3Pair],
    learning_rate: float,
    epochs: int,
) -> list[dict[str, float]]:
    """Train the model using gradient descent on softmax cross-entropy (context-3).

    Each example:
        context_ids = (token_id_{t-2}, token_id_{t-1}, token_id_t)
        target_id   = token_id_{t+1}

    Returns:
        A list of per-epoch metrics dictionaries (epoch, avg_loss, accuracy).
    """
    history: list[dict[str, float]] = []
    vocab_size: int = model.vocab_size

    for epoch in range(1, epochs + 1):
        total_loss: float = 0.0
        correct: int = 0

        for context_ids, target_id in pairs:
            previous2_id, previous1_id, current_id = context_ids

            # Forward pass: probabilities for next token given (t-2, t-1, t).
            probs: list[float] = model.forward(previous2_id, previous1_id, current_id)

            # Loss: how surprised is the model by the correct next token?
            loss: float = cross_entropy_loss(probs, target_id)
            total_loss += loss

            # Accuracy: did the top prediction match the target?
            pred_id: int = argmax(probs)
            if pred_id == target_id:
                correct += 1

            # Backward pass for softmax cross-entropy:
            #   grad[j] = probs[j] - y[j]  where y is one-hot(target_id)
            #
            # Update the weight row for this specific (t-2, t-1, t) context.
            row_idx: int = token_row_index_context3(context_ids, vocab_size=vocab_size)
            row: list[float] = model.weights[row_idx]

            for j in range(vocab_size):
                y: float = 1.0 if j == target_id else 0.0
                grad: float = probs[j] - y
                row[j] -= learning_rate * grad

        avg_loss: float = total_loss / len(pairs) if pairs else float("nan")
        accuracy: float = correct / len(pairs) if pairs else 0.0

        history.append(
            {
                "epoch": float(epoch),
                "avg_loss": avg_loss,
                "accuracy": accuracy,
            }
        )

        LOG.info(
            f"Epoch {epoch}/{epochs} | avg_loss={avg_loss:.6f} | accuracy={accuracy:.3f}"
        )

    return history


def main() -> None:
    """Run a simple training demo end-to-end (context-3)."""
    from toy_gpt_train.a_tokenizer import CORPUS_DIR, SimpleTokenizer
    from toy_gpt_train.b_vocab import Vocabulary

    log_header(LOG, "Training Demo: Next-Token Softmax Regression (Context-3)")
    log_path(LOG, "BASE_DIR", BASE_DIR)
    log_path(LOG, "OUTPUTS_DIR", OUTPUTS_DIR)
    log_path(LOG, "TRAIN_LOG_PATH", TRAIN_LOG_PATH)

    # Step 0: Identify the corpus file (single file rule).
    corpus_path: Path = find_single_corpus_file(CORPUS_DIR)

    # Step 1: Load and tokenize the corpus.
    tokenizer: SimpleTokenizer = SimpleTokenizer(corpus_path=corpus_path)
    tokens: list[str] = tokenizer.get_tokens()

    if len(tokens) < 4:
        LOG.error(
            "Need at least 4 tokens for context-3 training (t-2, t-1, t -> next)."
        )
        return

    # Step 2: Build vocabulary (maps tokens <-> integer IDs).
    vocab: Vocabulary = Vocabulary(tokens)
    vocab_size: int = vocab.vocab_size()

    # Step 3: Convert token strings to integer IDs for training.
    token_ids: list[int] = []
    for tok in tokens:
        tok_id: int | None = vocab.get_token_id(tok)
        if tok_id is None:
            LOG.error("Token not found in vocabulary: %r", tok)
            return
        token_ids.append(tok_id)

    # Step 4: Create training pairs (context-3 -> next).
    pairs: list[Context3Pair] = make_training_pairs(token_ids)
    LOG.info(f"Created {len(pairs)} training pairs.")

    # Step 5: Initialize model with zero weights (context-3 table lives in c_model.py).
    model: SimpleNextTokenModel = SimpleNextTokenModel(vocab_size=vocab_size)

    # Step 6: Train the model.
    learning_rate: float = 0.1
    epochs: int = 50

    history: list[dict[str, float]] = train_model(
        model=model,
        pairs=pairs,
        learning_rate=learning_rate,
        epochs=epochs,
    )

    # Step 7: Save training metrics for analysis.
    write_training_log(TRAIN_LOG_PATH, history)

    # Step 7b: Write inspectable artifacts for downstream use.
    write_artifacts(
        base_dir=BASE_DIR,
        corpus_path=corpus_path,
        vocab=vocab,
        model=model,
        model_kind="context3",
        learning_rate=learning_rate,
        epochs=epochs,
        row_labeler=row_labeler_context3(vocab, vocab_size),
    )

    # Step 8: Qualitative check - what does the model predict after the first 3 tokens?
    previous2_token: str = tokens[0]
    previous1_token: str = tokens[1]
    current_token: str = tokens[2]

    previous2_id: int | None = vocab.get_token_id(previous2_token)
    previous1_id: int | None = vocab.get_token_id(previous1_token)
    current_id: int | None = vocab.get_token_id(current_token)

    if previous2_id is None or previous1_id is None or current_id is None:
        LOG.error("One of the sample tokens was not found in vocabulary.")
        return

    probs: list[float] = model.forward(previous2_id, previous1_id, current_id)
    best_next_id: int = argmax(probs)
    best_next_tok: str | None = vocab.get_id_token(best_next_id)

    LOG.info(
        f"After training, most likely next token after {previous2_token!r}|{previous1_token!r}|{current_token!r} is {best_next_tok!r} (ID: {best_next_id})."
    )


if __name__ == "__main__":
    main()
