"""d_train.py - Training loop for the attention-based next-token model.

Trains AttentionNextTokenModel on a token corpus using a sliding context window.

Responsibilities:
- Create (context_ids -> next_token_id) training pairs
- Run gradient descent updating all parameter groups
- Track loss and accuracy per epoch
- Write a CSV log of training progress
- Write inspectable training artifacts

Concepts:
- backpropagation through attention: gradients flow from the output projection
  back through the attention weighted sum, then through Q/K/V projections,
  and finally into the embedding vectors.
- attention gradient: the softmax over attention scores has a Jacobian;
  the gradient through it uses the standard softmax backward formula.
- gradient clipping: caps individual gradient values before applying updates.
  Prevents weight explosion when gradients compound through multiple layers.
  Standard practice for attention and RNN training.

Key difference from embeddings (d_train.py in train-500-embeddings):
- Embeddings concatenate context vectors and pass through a single linear layer.
  Gradient flows back through W and into embeddings directly.
- Attention routes gradient through W_out, then back through the attention
  weighted sum (touching W_V), then through the attention score softmax
  (touching W_Q and W_K), then into embeddings.
  Six parameter groups are updated per example vs two in train-500.
"""

import logging
from pathlib import Path
from typing import Final

from datafun_toolkit.logger import get_logger, log_header, log_path

from toy_gpt_train.c_model import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_HEAD_DIM,
    AttentionNextTokenModel,
)
from toy_gpt_train.io_artifacts import (
    VocabularyLike,
    find_single_corpus_file,
    write_artifacts,
    write_training_log,
)
from toy_gpt_train.math_training import argmax, cross_entropy_loss, softmax

LOG: logging.Logger = get_logger("TRAIN", level="INFO")

BASE_DIR: Final[Path] = Path(__file__).resolve().parents[2]
OUTPUTS_DIR: Final[Path] = BASE_DIR / "outputs"
TRAIN_LOG_PATH: Final[Path] = OUTPUTS_DIR / "train_log.csv"

# WHY: Gradient clipping prevents weight explosion in attention training.
# Gradients compound through Q/K/V projections and the attention softmax;
# without clipping, a single large gradient can destroy all learned weights.
MAX_GRAD: Final[float] = 1.0

type ContextPair = tuple[list[int], int]


def make_training_pairs(
    token_ids: list[int],
    context_size: int,
) -> list[ContextPair]:
    """Convert token IDs into (context_ids, next_id) training pairs.

    Example (context_size=2):
        ids = [3, 1, 2, 4, 5]
        pairs = [([3, 1], 2), ([1, 2], 4), ([2, 4], 5)]
    """
    pairs: list[ContextPair] = []
    for i in range(len(token_ids) - context_size):
        context_ids = token_ids[i : i + context_size]
        next_id = token_ids[i + context_size]
        pairs.append((context_ids, next_id))
    return pairs


def row_labeler_attention(vocab: VocabularyLike, context_size: int):  # type: ignore[return]
    """Map a W_out row index to a label for artifact inspection."""

    def label(row_idx: int) -> str:
        return f"head_dim_{row_idx}"

    return label


def _clip(val: float) -> float:
    """Clip a gradient value to [-MAX_GRAD, MAX_GRAD]."""
    return max(-MAX_GRAD, min(MAX_GRAD, val))


def _softmax_backward(weights: list[float], d_out: list[float]) -> list[float]:
    """Backward pass through softmax.

    Given softmax output weights and upstream gradient d_out,
    compute the gradient w.r.t. the pre-softmax scores.

    Formula (Jacobian of softmax):
        d_scores[i] = weights[i] * (d_out[i] - sum_j(weights[j] * d_out[j]))

    Args:
        weights: Softmax output (attention weights), length n.
        d_out:   Upstream gradient, length n.

    Returns:
        Gradient w.r.t. pre-softmax scores, length n.
    """
    dot = sum(weights[j] * d_out[j] for j in range(len(weights)))
    return [weights[i] * (d_out[i] - dot) for i in range(len(weights))]


def train_model(
    model: AttentionNextTokenModel,
    pairs: list[ContextPair],
    learning_rate: float,
    epochs: int,
) -> list[dict[str, float]]:
    """Train the model with gradient descent on softmax cross-entropy.

    Six parameter groups are updated per example:
    1. W_out  (output projection)
    2. bias
    3. W_V    (value projection)
    4. W_Q    (query projection)
    5. W_K    (key projection)
    6. embeddings + pos_embeddings

    Gradient derivation (single-head attention, last position predicts):

        Forward (last position i = context_size - 1):
            e_t   = embeddings[token_id] + pos_embeddings[t]  for each t
            Q_t   = e_t @ W_Q
            K_t   = e_t @ W_K
            V_t   = e_t @ W_V
            s_j   = dot(Q_i, K_j) * scale         attention scores
            a_j   = softmax(s)[j]                  attention weights
            ctx   = sum_j(a_j * V_j)              context vector
            out   = ctx @ W_out + bias
            probs = softmax(out)

        Loss gradient w.r.t. output scores:
            d_out[j] = probs[j] - y[j]             (softmax cross-entropy shortcut)

        W_out gradient:
            d_W_out[k][j] = ctx[k] * d_out[j]

        bias gradient:
            d_bias[j] = d_out[j]

        Gradient w.r.t. ctx (context vector):
            d_ctx[k] = sum_j(W_out[k][j] * d_out[j])

        Gradient w.r.t. V_j (value vectors):
            d_V[j][k] = a_j * d_ctx[k]

        Gradient w.r.t. attention weights a_j:
            d_a[j] = dot(d_ctx, V_j)

        Gradient w.r.t. attention scores s_j (through attention softmax):
            d_s = softmax_backward(a, d_a)

        Gradient w.r.t. Q_i and K_j (last query position only):
            d_Q_i[k] = sum_j(d_s[j] * K_j[k]) * scale
            d_K_j[k] = d_s[j] * Q_i[k] * scale

        W_Q, W_K, W_V gradients (for each context position t):
            d_W_V[m][k] += e_t[m] * d_V[t][k]
            d_W_Q[m][k] += e_t[m] * d_Q_t[k]   (only last position i)
            d_W_K[m][k] += e_t[m] * d_K_t[k]

        Embedding gradient (for each context position t):
            d_E[token_t][m] += sum_k(W_V[m][k] * d_V[t][k])
                              + sum_k(W_Q[m][k] * d_Q_t[k])   (only last position)
                              + sum_k(W_K[m][k] * d_K_t[k])

        Positional embedding gradient (same as token embedding gradient):
            d_pos[t][m] = d_E[token_t][m]

    Args:
        model:         The AttentionNextTokenModel to train.
        pairs:         Training pairs (context_ids, next_id).
        learning_rate: Step size for gradient updates.
        epochs:        Number of full passes through the training data.

    Returns:
        List of per-epoch metric dictionaries (epoch, avg_loss, accuracy).
    """
    history: list[dict[str, float]] = []
    vocab_size = model.vocab_size
    embedding_dim = model.embedding_dim
    head_dim = model.head_dim
    context_size = model.context_size
    scale = model.scale

    LOG.info(
        f"Context size: {context_size}, embedding dim: {embedding_dim}, "
        f"head_dim: {head_dim}, vocab size: {vocab_size}, "
        f"grad_clip={MAX_GRAD}"
    )

    for epoch in range(1, epochs + 1):
        total_loss: float = 0.0
        correct: int = 0

        for context_ids, target_id in pairs:
            # ============================================================
            # FORWARD PASS
            # ============================================================

            # 1. Embedding lookup with positional encoding.
            embs: list[list[float]] = [
                [
                    model.embeddings[tid][k] + model.pos_embeddings[pos][k]
                    for k in range(embedding_dim)
                ]
                for pos, tid in enumerate(context_ids)
            ]

            # 2. Q, K, V projections for all context positions.
            Qs: list[list[float]] = [model._project(e, model.W_Q) for e in embs]
            Ks: list[list[float]] = [model._project(e, model.W_K) for e in embs]
            Vs: list[list[float]] = [model._project(e, model.W_V) for e in embs]

            # 3. Attention scores and weights (using last position as query).
            i_last = context_size - 1
            attn_scores: list[float] = [
                model._dot(Qs[i_last], Ks[j]) * scale for j in range(context_size)
            ]
            attn_weights: list[float] = softmax(attn_scores)

            # 4. Context vector: weighted sum of values.
            ctx: list[float] = [0.0] * head_dim
            for j in range(context_size):
                for k in range(head_dim):
                    ctx[k] += attn_weights[j] * Vs[j][k]

            # 5. Output projection -> vocab scores.
            out_scores: list[float] = list(model.bias)
            for k, val in enumerate(ctx):
                for j in range(vocab_size):
                    out_scores[j] += val * model.W_out[k][j]

            # 6. Softmax -> probabilities.
            probs: list[float] = softmax(out_scores)

            # ============================================================
            # LOSS AND ACCURACY
            # ============================================================

            total_loss += cross_entropy_loss(probs, target_id)
            if argmax(probs) == target_id:
                correct += 1

            # ============================================================
            # BACKWARD PASS
            # ============================================================

            # --- Output layer ---

            # d_out[j] = probs[j] - y[j]
            d_out: list[float] = [
                probs[j] - (1.0 if j == target_id else 0.0) for j in range(vocab_size)
            ]

            # d_W_out[k][j] = ctx[k] * d_out[j]
            for k in range(head_dim):
                for j in range(vocab_size):
                    model.W_out[k][j] -= learning_rate * _clip(ctx[k] * d_out[j])

            # d_bias[j] = d_out[j]
            for j in range(vocab_size):
                model.bias[j] -= learning_rate * _clip(d_out[j])

            # --- Context vector gradient ---

            # d_ctx[k] = sum_j(W_out[k][j] * d_out[j])
            d_ctx: list[float] = [
                sum(model.W_out[k][j] * d_out[j] for j in range(vocab_size))
                for k in range(head_dim)
            ]

            # --- Value gradients ---

            # d_V[j][k] = attn_weights[j] * d_ctx[k]
            d_Vs: list[list[float]] = [
                [attn_weights[j] * d_ctx[k] for k in range(head_dim)]
                for j in range(context_size)
            ]

            # --- Attention weight gradients ---

            # d_a[j] = dot(d_ctx, V_j)
            d_attn_weights: list[float] = [
                sum(d_ctx[k] * Vs[j][k] for k in range(head_dim))
                for j in range(context_size)
            ]

            # --- Attention score gradients (through attention softmax) ---

            d_attn_scores: list[float] = _softmax_backward(attn_weights, d_attn_weights)

            # Scale the attention score gradients.
            d_attn_scores = [s * scale for s in d_attn_scores]

            # --- Q and K gradients (last query position only) ---

            # d_Q_i[k] = sum_j(d_attn_scores[j] * K_j[k])
            d_Q_last: list[float] = [
                sum(d_attn_scores[j] * Ks[j][k] for j in range(context_size))
                for k in range(head_dim)
            ]

            # d_K_j[k] = d_attn_scores[j] * Q_i[k]
            d_Ks: list[list[float]] = [
                [d_attn_scores[j] * Qs[i_last][k] for k in range(head_dim)]
                for j in range(context_size)
            ]

            # --- W_Q, W_K, W_V gradients and embedding gradients ---

            for t in range(context_size):
                token_id = context_ids[t]
                e_t = embs[t]

                # W_V: d_W_V[m][k] += e_t[m] * d_V[t][k]
                for m in range(embedding_dim):
                    for k in range(head_dim):
                        model.W_V[m][k] -= learning_rate * _clip(e_t[m] * d_Vs[t][k])

                # W_K: d_W_K[m][k] += e_t[m] * d_K_t[k]
                for m in range(embedding_dim):
                    for k in range(head_dim):
                        model.W_K[m][k] -= learning_rate * _clip(e_t[m] * d_Ks[t][k])

                # W_Q: only the last position contributes to Q gradient.
                if t == i_last:
                    for m in range(embedding_dim):
                        for k in range(head_dim):
                            model.W_Q[m][k] -= learning_rate * _clip(
                                e_t[m] * d_Q_last[k]
                            )

                # Embedding gradient: sum contributions from V, K, and Q paths.
                d_emb: list[float] = [0.0] * embedding_dim

                # From V path: d_E[m] += sum_k(W_V[m][k] * d_V[t][k])
                for m in range(embedding_dim):
                    d_emb[m] += sum(
                        model.W_V[m][k] * d_Vs[t][k] for k in range(head_dim)
                    )

                # From K path: d_E[m] += sum_k(W_K[m][k] * d_K_t[k])
                for m in range(embedding_dim):
                    d_emb[m] += sum(
                        model.W_K[m][k] * d_Ks[t][k] for k in range(head_dim)
                    )

                # From Q path (last position only).
                if t == i_last:
                    for m in range(embedding_dim):
                        d_emb[m] += sum(
                            model.W_Q[m][k] * d_Q_last[k] for k in range(head_dim)
                        )

                # Apply token embedding update with clipping.
                for m in range(embedding_dim):
                    model.embeddings[token_id][m] -= learning_rate * _clip(d_emb[m])

                # Apply positional embedding update with clipping.
                # WHY: pos_embeddings receive the same gradient as token embeddings
                # because they are added together before Q/K/V projection.
                for m in range(embedding_dim):
                    model.pos_embeddings[t][m] -= learning_rate * _clip(d_emb[m])

        avg_loss = total_loss / len(pairs) if pairs else float("nan")
        accuracy = correct / len(pairs) if pairs else 0.0

        history.append(
            {"epoch": float(epoch), "avg_loss": avg_loss, "accuracy": accuracy}
        )
        LOG.info(
            f"Epoch {epoch}/{epochs} | avg_loss={avg_loss:.6f} | accuracy={accuracy:.3f}"
        )

    return history


def main() -> None:
    """Run attention model training end-to-end."""
    from toy_gpt_train.a_tokenizer import CORPUS_DIR, SimpleTokenizer
    from toy_gpt_train.b_vocab import Vocabulary

    log_header(LOG, "Training Demo: Next-Token Prediction with Attention")
    log_path(LOG, "BASE_DIR", BASE_DIR)
    log_path(LOG, "OUTPUTS_DIR", OUTPUTS_DIR)

    corpus_path: Path = find_single_corpus_file(CORPUS_DIR)

    tokenizer: SimpleTokenizer = SimpleTokenizer(corpus_path=corpus_path)
    tokens: list[str] = tokenizer.get_tokens()

    if len(tokens) < DEFAULT_CONTEXT_SIZE + 1:
        LOG.error(f"Need at least {DEFAULT_CONTEXT_SIZE + 1} tokens for training.")
        return

    vocab: Vocabulary = Vocabulary(tokens)
    vocab_size: int = vocab.vocab_size()

    token_ids: list[int] = []
    for tok in tokens:
        tok_id = vocab.get_token_id(tok)
        if tok_id is None:
            LOG.error("Token not found in vocabulary: %r", tok)
            return
        token_ids.append(tok_id)

    pairs: list[ContextPair] = make_training_pairs(
        token_ids, context_size=DEFAULT_CONTEXT_SIZE
    )
    LOG.info(
        f"Created {len(pairs)} training pairs (context_size={DEFAULT_CONTEXT_SIZE})."
    )

    model: AttentionNextTokenModel = AttentionNextTokenModel(
        vocab_size=vocab_size,
        embedding_dim=DEFAULT_EMBEDDING_DIM,
        head_dim=DEFAULT_HEAD_DIM,
        context_size=DEFAULT_CONTEXT_SIZE,
    )

    # OBS: lr=0.01 caused no symmetry breaking without positional embeddings.
    # OBS: positional embeddings added to c_model.py to break position symmetry.
    # OBS: gradient clipping (MAX_GRAD=1.0) added to prevent weight explosion.
    learning_rate: float = 0.01
    epochs: int = 100

    history = train_model(
        model=model, pairs=pairs, learning_rate=learning_rate, epochs=epochs
    )

    write_training_log(TRAIN_LOG_PATH, history)

    write_artifacts(
        base_dir=BASE_DIR,
        corpus_path=corpus_path,
        vocab=vocab,
        model=model,
        model_kind="attention",
        learning_rate=learning_rate,
        epochs=epochs,
        row_labeler=row_labeler_attention(vocab, DEFAULT_CONTEXT_SIZE),
    )

    # Qualitative check.
    context_tokens = tokens[:DEFAULT_CONTEXT_SIZE]
    context_ids: list[int] = []
    for tok in context_tokens:
        tid = vocab.get_token_id(tok)
        if tid is None:
            LOG.error("Token not found: %r", tok)
            return
        context_ids.append(tid)

    probs = model.forward(context_ids)
    best_id = argmax(probs)
    best_tok = vocab.get_id_token(best_id)

    LOG.info(
        f"After training, most likely next token after "
        f"{context_tokens} is {best_tok!r} (ID {best_id})."
    )

    # Show attention weights after training — the inspectable artifact
    # unique to this model.
    attn = model.get_attention_weights(context_ids)
    LOG.info(f"Attention weights after training (context: {context_tokens}):")
    for i, row in enumerate(attn):
        tok = context_tokens[i]
        formatted = "  ".join(f"{w:.3f}" for w in row)
        LOG.info(f"  {tok!r}: [{formatted}]")


if __name__ == "__main__":
    main()
