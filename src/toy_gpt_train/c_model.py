"""c_model.py - Attention-based next-token prediction model.

Defines a next-token prediction model that uses single-head self-attention
to dynamically weight which context tokens matter most for each prediction.

Concepts:
- query (Q): a projection of the current token asking "what am I looking for?"
- key (K):   a projection of each context token saying "what do I contain?"
- value (V): a projection of each context token saying "what do I contribute?"
- attention score: dot product of a query with a key, scaled by sqrt(head_dim).
  High score means the query and key are well-aligned.
- attention weights: softmax over attention scores — a probability distribution
  over context positions indicating how much to attend to each.
- context vector: weighted sum of value vectors using attention weights.

Architecture:
    token IDs -> embedding lookup -> Q/K/V projections -> scaled dot-product
    attention -> context vector -> linear -> softmax -> probs

Key departure from embeddings (train-500):
- Embeddings concatenate all context vectors and pass them through a fixed
  linear layer. Every context position is weighted equally.
- Attention learns to weight context positions dynamically per prediction.
  The model can learn "for this query token, the token two steps back
  matters more than the one immediately before."

Reference: Vaswani et al., "Attention Is All You Need" (2017).
    https://arxiv.org/abs/1706.03762

Training is handled in d_train.py.
"""

import logging
import math
import random
from typing import Final

from datafun_toolkit.logger import get_logger, log_header

from toy_gpt_train.math_training import softmax

LOG: logging.Logger = get_logger("MODEL", level="INFO")

DEFAULT_EMBEDDING_DIM: Final[int] = 16
DEFAULT_HEAD_DIM: Final[int] = 16
DEFAULT_CONTEXT_SIZE: Final[int] = 2


class AttentionNextTokenModel:
    """Next-token prediction model using single-head self-attention.

    Parameters:
        vocab_size:     Number of unique tokens in the vocabulary.
        embedding_dim:  Size of each token's embedding vector.
        head_dim:       Size of the query, key, and value projections.
        context_size:   Number of preceding tokens used as context.

    Parameter groups:
        embeddings  vocab_size x embedding_dim
        W_Q         embedding_dim x head_dim   (query projection)
        W_K         embedding_dim x head_dim   (key projection)
        W_V         embedding_dim x head_dim   (value projection)
        W_out       head_dim x vocab_size      (output projection)
        bias        vocab_size
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        head_dim: int = DEFAULT_HEAD_DIM,
        context_size: int = DEFAULT_CONTEXT_SIZE,
    ) -> None:
        """Initialize all model parameters with small random values."""
        self.vocab_size: Final[int] = vocab_size
        self.embedding_dim: Final[int] = embedding_dim
        self.head_dim: Final[int] = head_dim
        self.context_size: Final[int] = context_size
        self.scale: Final[float] = 1.0 / math.sqrt(head_dim)

        # Embedding matrix: vocab_size x embedding_dim
        self.embeddings: list[list[float]] = [
            [random.gauss(0.0, 0.01) for _ in range(embedding_dim)]
            for _ in range(vocab_size)
        ]

        # Additional positional embeddings
        self.pos_embeddings: list[list[float]] = [
            [random.gauss(0.0, 0.01) for _ in range(embedding_dim)]
            for _ in range(context_size)
        ]

        # Q, K, V projection matrices: embedding_dim x head_dim
        self.W_Q: list[list[float]] = [
            [random.gauss(0.0, 0.01) for _ in range(head_dim)]
            for _ in range(embedding_dim)
        ]
        self.W_K: list[list[float]] = [
            [random.gauss(0.0, 0.01) for _ in range(head_dim)]
            for _ in range(embedding_dim)
        ]
        self.W_V: list[list[float]] = [
            [random.gauss(0.0, 0.01) for _ in range(head_dim)]
            for _ in range(embedding_dim)
        ]

        # Output projection: head_dim x vocab_size
        self.W_out: list[list[float]] = [
            [random.gauss(0.0, 0.01) for _ in range(vocab_size)]
            for _ in range(head_dim)
        ]

        # Bias: one value per output token
        self.bias: list[float] = [0.0] * vocab_size

        # io_artifacts protocol compatibility
        self.weights: list[list[float]] = self.W_out

        n_embed = vocab_size * embedding_dim
        n_qkv = 3 * embedding_dim * head_dim
        n_out = head_dim * vocab_size
        n_bias = vocab_size

        LOG.info(
            f"AttentionNextTokenModel initialized: "
            f"vocab_size={vocab_size}, "
            f"embedding_dim={embedding_dim}, "
            f"head_dim={head_dim}, "
            f"context_size={context_size}."
        )

        n_pos = context_size * embedding_dim

        LOG.info(
            f"Parameters: "
            f"embeddings={n_embed:,}, "
            f"pos_embeddings={n_pos:,}, "
            f"Q+K+V={n_qkv:,}, "
            f"W_out={n_out:,}, "
            f"bias={n_bias} "
            f"(total={n_embed + n_pos + n_qkv + n_out + n_bias:,})."
        )

    # ============================================================
    # Linear projection helpers
    # ============================================================

    @staticmethod
    def _project(vec: list[float], matrix: list[list[float]]) -> list[float]:
        """Project a vector through a matrix: result[j] = sum_i(vec[i] * matrix[i][j]).

        Args:
            vec:    Input vector of length m.
            matrix: Weight matrix of shape m x n.

        Returns:
            Output vector of length n.
        """
        n = len(matrix[0])
        result = [0.0] * n
        for i, val in enumerate(vec):
            for j in range(n):
                result[j] += val * matrix[i][j]
        return result

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        """Dot product of two equal-length vectors."""
        return sum(x * y for x, y in zip(a, b, strict=False))

    # ============================================================
    # Attention
    # ============================================================

    def _attention(
        self,
        queries: list[list[float]],
        keys: list[list[float]],
        values: list[list[float]],
    ) -> list[list[float]]:
        """Compute scaled dot-product attention.

        For each query position i, compute a weighted sum of value vectors
        where the weights are determined by how well query i matches each key j.

        Formula:
            scores[i][j] = dot(Q[i], K[j]) * scale
            weights[i]   = softmax(scores[i])
            output[i]    = sum_j( weights[i][j] * V[j] )

        Args:
            queries: List of query vectors, one per context position.
            keys:    List of key vectors, one per context position.
            values:  List of value vectors, one per context position.

        Returns:
            List of output vectors (one per query position), each of length head_dim.
        """
        n = len(queries)
        outputs: list[list[float]] = []

        for i in range(n):
            # Compute raw attention scores for query i against all keys.
            scores: list[float] = [
                self._dot(queries[i], keys[j]) * self.scale for j in range(n)
            ]

            # Convert to weights via softmax.
            weights: list[float] = softmax(scores)

            # Weighted sum of value vectors.
            out: list[float] = [0.0] * self.head_dim
            for j in range(n):
                for k in range(self.head_dim):
                    out[k] += weights[j] * values[j][k]

            outputs.append(out)

        return outputs

    # ============================================================
    # Forward pass
    # ============================================================

    def forward(self, context_ids: list[int]) -> list[float]:
        """Compute next-token probabilities using self-attention.

        Steps:
            1. Look up embedding for each context token.
            2. Project each embedding into Q, K, V spaces.
            3. Run scaled dot-product attention.
            4. Take the output at the last context position.
            5. Project through W_out + bias -> vocab scores.
            6. Softmax -> probabilities.

        Args:
            context_ids: Token IDs of length context_size.

        Returns:
            Probability distribution over the vocabulary.

        Raises:
            ValueError: If context_ids length does not match context_size.
        """
        if len(context_ids) != self.context_size:
            msg = f"Expected {self.context_size} context IDs, got {len(context_ids)}."
            raise ValueError(msg)

        # Step 1: Embedding lookup.
        # embs: list[list[float]] = [self.embeddings[tid] for tid in context_ids]
        embs: list[list[float]] = [
            [
                self.embeddings[tid][k] + self.pos_embeddings[pos][k]
                for k in range(self.embedding_dim)
            ]
            for pos, tid in enumerate(context_ids)
        ]

        # Step 2: Project into Q, K, V.
        queries = [self._project(e, self.W_Q) for e in embs]
        keys = [self._project(e, self.W_K) for e in embs]
        values = [self._project(e, self.W_V) for e in embs]

        # Step 3: Self-attention.
        attended = self._attention(queries, keys, values)

        # Step 4: Use last position's output as the prediction vector.
        ctx_vec = attended[-1]

        # Step 5: Output projection -> vocab scores.
        scores: list[float] = list(self.bias)
        for i, val in enumerate(ctx_vec):
            for j in range(self.vocab_size):
                scores[j] += val * self.W_out[i][j]

        # Step 6: Softmax -> probabilities.
        return softmax(scores)

    def get_attention_weights(self, context_ids: list[int]) -> list[list[float]]:
        """Return the attention weight matrix for a context window.

        Useful for inspection and visualization — shows which context
        positions each query position attends to most.

        Args:
            context_ids: Token IDs of length context_size.

        Returns:
            Matrix of shape context_size x context_size.
            Row i, column j = how much position i attends to position j.
        """
        if len(context_ids) != self.context_size:
            msg = f"Expected {self.context_size} context IDs, got {len(context_ids)}."
            raise ValueError(msg)

        embs = [
            [
                self.embeddings[tid][k] + self.pos_embeddings[pos][k]
                for k in range(self.embedding_dim)
            ]
            for pos, tid in enumerate(context_ids)
        ]
        queries = [self._project(e, self.W_Q) for e in embs]
        keys = [self._project(e, self.W_K) for e in embs]

        n = len(queries)
        weight_matrix: list[list[float]] = []
        for i in range(n):
            scores = [self._dot(queries[i], keys[j]) * self.scale for j in range(n)]
            weight_matrix.append(softmax(scores))

        return weight_matrix


def main() -> None:
    """Demonstrate a forward pass and attention weights (untrained model)."""
    from toy_gpt_train.a_tokenizer import SimpleTokenizer
    from toy_gpt_train.b_vocab import Vocabulary

    log_header(LOG, "Attention Next-Token Model Demo")

    tokenizer: SimpleTokenizer = SimpleTokenizer()
    tokens: list[str] = tokenizer.get_tokens()

    if len(tokens) < DEFAULT_CONTEXT_SIZE + 1:
        LOG.info(f"Need at least {DEFAULT_CONTEXT_SIZE + 1} tokens for demonstration.")
        return

    vocab: Vocabulary = Vocabulary(tokens)
    model: AttentionNextTokenModel = AttentionNextTokenModel(
        vocab_size=vocab.vocab_size(),
        embedding_dim=DEFAULT_EMBEDDING_DIM,
        head_dim=DEFAULT_HEAD_DIM,
        context_size=DEFAULT_CONTEXT_SIZE,
    )

    context_tokens = tokens[:DEFAULT_CONTEXT_SIZE]
    context_ids: list[int] = []
    for tok in context_tokens:
        tid = vocab.get_token_id(tok)
        if tid is None:
            LOG.info(f"Token {tok!r} not found in vocabulary.")
            return
        context_ids.append(tid)

    probs: list[float] = model.forward(context_ids)
    attn: list[list[float]] = model.get_attention_weights(context_ids)

    LOG.info(f"Context tokens: {context_tokens}")
    LOG.info(f"Context IDs:    {context_ids}")
    LOG.info("Top 5 predicted next tokens (untrained — random weights):")
    indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    for rank, (idx, prob) in enumerate(indexed[:5], 1):
        tok = vocab.get_id_token(idx)
        LOG.info(f"  {rank}. {tok!r} (ID {idx}) -> {prob:.4f}")

    LOG.info("Attention weights (row i attends to column j):")
    for i, row in enumerate(attn):
        tok = context_tokens[i]
        formatted = "  ".join(f"{w:.3f}" for w in row)
        LOG.info(f"  {tok!r}: [{formatted}]")


if __name__ == "__main__":
    main()
