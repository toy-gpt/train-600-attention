"""e_infer.py - Inference module (artifact-driven, attention model).

Runs inference using previously saved training artifacts.

Responsibilities:
- Load training artifacts from artifacts/
  - 00_meta.json
  - 01_vocabulary.csv
  - 02_model_weights.csv  (W_out: head_dim x vocab_size)
  - 03_token_embeddings.csv  (learned token embedding vectors)
  - 04_positional_embeddings.csv  (learned positional embedding vectors)
- Reconstruct the model and vocabulary from artifacts
- Generate tokens using greedy decoding (argmax)
- Print top-k next-token probabilities for inspection
- Show attention weights for the seed context

Notes:
- This module does NOT retrain.
- If artifacts are missing, run d_train.py first.
- Bootstrapping: generation starts from context_size seed tokens.
  If fewer seeds are provided, the most common token is repeated to fill the window.
"""

import csv
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import tomllib
from typing import Final

from datafun_toolkit.logger import get_logger, log_header

from toy_gpt_train.c_model import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_HEAD_DIM,
    AttentionNextTokenModel,
)
from toy_gpt_train.math_training import argmax

LOG: logging.Logger = get_logger("INFER", level="INFO")

BASE_DIR: Final[Path] = Path(__file__).resolve().parents[2]
CONFIG_PATH: Final[Path] = BASE_DIR / "config.toml"
ARTIFACTS_DIR: Final[Path] = BASE_DIR / "artifacts"
META_PATH: Final[Path] = ARTIFACTS_DIR / "00_meta.json"
VOCAB_PATH: Final[Path] = ARTIFACTS_DIR / "01_vocabulary.csv"
WEIGHTS_PATH: Final[Path] = ARTIFACTS_DIR / "02_model_weights.csv"
EMBEDDINGS_PATH: Final[Path] = ARTIFACTS_DIR / "03_token_embeddings.csv"
POS_EMBEDDINGS_PATH: Final[Path] = ARTIFACTS_DIR / "04_positional_embeddings.csv"

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]


# ============================================================
# Vocabulary
# ============================================================


@dataclass(frozen=True)
class ArtifactVocabulary:
    """Vocabulary reconstructed from artifacts/01_vocabulary.csv."""

    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    token_freq: dict[str, int]

    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def get_token_id(self, token: str) -> int | None:
        return self.token_to_id.get(token)

    def get_id_token(self, idx: int) -> str | None:
        return self.id_to_token.get(idx)

    def get_token_frequency(self, token: str) -> int:
        return self.token_freq.get(token, 0)


# ============================================================
# Artifact loaders
# ============================================================


def require_artifacts() -> None:
    """Fail fast with a helpful message if artifacts are missing."""
    missing = [
        p
        for p in [
            META_PATH,
            VOCAB_PATH,
            WEIGHTS_PATH,
            EMBEDDINGS_PATH,
            POS_EMBEDDINGS_PATH,
        ]
        if not p.exists()
    ]
    if missing:
        LOG.error("Missing training artifacts:")
        for p in missing:
            LOG.error(f"  - {p}")
        LOG.error("Run training first:  uv run python src/toy_gpt_train/d_train.py")
        raise SystemExit(2)


def load_meta(path: Path) -> JsonObject:
    with path.open("r", encoding="utf-8") as f:
        data: JsonObject = json.load(f)
    return data


def load_config() -> JsonObject:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("rb") as f:
            return tomllib.load(f)
    return {}


def load_vocabulary_csv(path: Path) -> ArtifactVocabulary:
    token_to_id: dict[str, int] = {}
    id_to_token: dict[int, str] = {}
    token_freq: dict[str, int] = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"token_id", "token", "frequency"}
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise ValueError(
                f"Unexpected vocabulary header. Expected {sorted(expected)} "
                f"but got {reader.fieldnames}"
            )
        for row in reader:
            token_id = int(row["token_id"])
            token = row["token"]
            freq = int(row["frequency"])
            token_to_id[token] = token_id
            id_to_token[token_id] = token
            token_freq[token] = freq

    return ArtifactVocabulary(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        token_freq=token_freq,
    )


def load_w_out_csv(path: Path, head_dim: int, vocab_size: int) -> list[list[float]]:
    """Load 02_model_weights.csv -> W_out matrix.

    Expected shape: head_dim rows x vocab_size columns.
    """
    weights: list[list[float]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError("Weights CSV is empty.")
        num_outputs = len(header) - 1
        if num_outputs != vocab_size:
            raise ValueError(
                f"Weights CSV output width mismatch. "
                f"Expected {vocab_size} columns but found {num_outputs}."
            )
        for row in reader:
            if not row:
                continue
            weights.append([float(x) for x in row[1:]])

    if len(weights) != head_dim:
        raise ValueError(
            f"Weights CSV row count mismatch. "
            f"Expected {head_dim} rows but found {len(weights)}."
        )
    return weights


def load_token_embeddings_csv(
    path: Path, vocab_size: int, embedding_dim: int
) -> list[list[float]]:
    """Load 03_token_embeddings.csv -> learned token embedding matrix.

    Expected shape: vocab_size rows x embedding_dim columns.
    """
    embeddings: list[list[float]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Embeddings CSV is empty.")
        dim_cols = [c for c in reader.fieldnames if c.startswith("dim_")]
        if len(dim_cols) != embedding_dim:
            raise ValueError(
                f"Embeddings CSV dimension mismatch. "
                f"Expected {embedding_dim} dim columns but found {len(dim_cols)}."
            )
        for row in reader:
            embeddings.append([float(row[c]) for c in dim_cols])

    if len(embeddings) != vocab_size:
        raise ValueError(
            f"Embeddings CSV row count mismatch. "
            f"Expected {vocab_size} rows but found {len(embeddings)}."
        )
    return embeddings


def load_positional_embeddings_csv(
    path: Path, context_size: int, embedding_dim: int
) -> list[list[float]]:
    """Load 04_positional_embeddings.csv -> learned positional embedding matrix.

    Expected shape: context_size rows x embedding_dim columns.
    """
    pos_embeddings: list[list[float]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Positional embeddings CSV is empty.")
        dim_cols = [c for c in reader.fieldnames if c.startswith("dim_")]
        if len(dim_cols) != embedding_dim:
            raise ValueError(
                f"Positional embeddings CSV dimension mismatch. "
                f"Expected {embedding_dim} dim columns but found {len(dim_cols)}."
            )
        for row in reader:
            pos_embeddings.append([float(row[c]) for c in dim_cols])

    if len(pos_embeddings) != context_size:
        raise ValueError(
            f"Positional embeddings CSV row count mismatch. "
            f"Expected {context_size} rows but found {len(pos_embeddings)}."
        )
    return pos_embeddings


# ============================================================
# Inference
# ============================================================


def top_k(probs: list[float], k: int) -> list[tuple[int, float]]:
    """Return top-k (token_id, probability) pairs sorted by probability."""
    pairs = list(enumerate(probs))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:k]


def generate_tokens(
    model: AttentionNextTokenModel,
    vocab: ArtifactVocabulary,
    seed_tokens: list[str],
    num_tokens: int,
) -> list[str]:
    """Generate tokens using greedy decoding (argmax) with a sliding context window.

    Args:
        model:        Trained AttentionNextTokenModel.
        vocab:        ArtifactVocabulary loaded from artifacts.
        seed_tokens:  Starting tokens (must equal model.context_size).
        num_tokens:   Number of new tokens to generate.

    Returns:
        seed_tokens extended by num_tokens generated tokens.
    """
    context_size = model.context_size
    LOG.info(f"Generating {num_tokens} tokens with context size {context_size}...")
    generated: list[str] = list(seed_tokens)

    context_ids: list[int] = []
    for tok in seed_tokens:
        tid = vocab.get_token_id(tok)
        if tid is None:
            LOG.error(f"Seed token {tok!r} not in vocabulary.")
            return generated
        context_ids.append(tid)

    for _ in range(num_tokens):
        probs: list[float] = model.forward(context_ids)
        next_id: int = argmax(probs)
        next_token = vocab.get_id_token(next_id)

        if next_token is None:
            LOG.error(f"Generated invalid token ID: {next_id}")
            break

        generated.append(next_token)
        context_ids = context_ids[1:] + [next_id]  # slide window

    return generated


# ============================================================
# Main
# ============================================================


def main() -> None:
    """Run inference using saved training artifacts."""
    log_header(LOG, "Inference Demo: Attention Model (Load Artifacts and Generate)")

    require_artifacts()

    meta = load_meta(META_PATH)
    vocab = load_vocabulary_csv(VOCAB_PATH)
    config: JsonObject = load_config()
    infer_config: JsonObject = (
        config.get("infer", {})  # type: ignore[assignment]
        if isinstance(config.get("infer"), dict)
        else {}
    )

    v = vocab.vocab_size()
    embedding_dim: int = int(infer_config.get("embedding_dim", DEFAULT_EMBEDDING_DIM))  # type: ignore[arg-type]
    head_dim: int = int(infer_config.get("head_dim", DEFAULT_HEAD_DIM))  # type: ignore[arg-type]
    context_size: int = int(infer_config.get("context_size", DEFAULT_CONTEXT_SIZE))  # type: ignore[arg-type]
    num_tokens: int = int(infer_config.get("num_tokens", 10))  # type: ignore[arg-type]
    topk: int = int(infer_config.get("topk", 5))  # type: ignore[arg-type]

    # Load and reconstruct model from artifacts.
    model = AttentionNextTokenModel(
        vocab_size=v,
        embedding_dim=embedding_dim,
        head_dim=head_dim,
        context_size=context_size,
    )
    model.W_out = load_w_out_csv(WEIGHTS_PATH, head_dim=head_dim, vocab_size=v)
    model.weights = model.W_out
    model.embeddings = load_token_embeddings_csv(
        EMBEDDINGS_PATH, vocab_size=v, embedding_dim=embedding_dim
    )
    model.pos_embeddings = load_positional_embeddings_csv(
        POS_EMBEDDINGS_PATH, context_size=context_size, embedding_dim=embedding_dim
    )

    LOG.info(
        f"Loaded repo_name={meta.get('repo_name')} model_kind={meta.get('model_kind')}"
    )
    LOG.info(
        f"Vocab size: {v} | embedding_dim: {embedding_dim} | "
        f"head_dim: {head_dim} | context_size: {context_size}"
    )

    # Build seed from config; fall back to most common token repeated.
    most_common = (
        max(vocab.token_freq, key=lambda t: vocab.token_freq[t])
        if vocab.token_freq
        else "<no_tokens>"
    )
    seed_tokens: list[str] = []
    for i in range(context_size):
        tok = str(infer_config.get(f"seed_{i}", ""))
        seed_tokens.append(tok if tok else most_common)

    if any(not str(infer_config.get(f"seed_{i}", "")) for i in range(context_size)):
        LOG.warning(
            f"Seed tokens missing in config.toml; using {seed_tokens}. "
            "Predictions may be less meaningful."
        )

    LOG.info(f"Seed context: {seed_tokens}")

    # Resolve seed IDs.
    seed_ids: list[int] = []
    for tok in seed_tokens:
        tid = vocab.get_token_id(tok)
        if tid is None:
            LOG.error(f"Seed token {tok!r} not in vocabulary.")
            return
        seed_ids.append(tid)

    # Show top-k predictions for the seed context.
    probs = model.forward(seed_ids)
    LOG.info(f"Top-{topk} next-token predictions after {seed_tokens}:")
    for tok_id, prob in top_k(probs, k=topk):
        tok = vocab.get_id_token(tok_id)
        LOG.info(f"  {tok!r} (ID {tok_id}): {prob:.4f}")

    # Show attention weights, inspectable artifact unique to this model.
    attn = model.get_attention_weights(seed_ids)
    LOG.info(f"Attention weights for seed context {seed_tokens}:")
    for i, row in enumerate(attn):
        tok = seed_tokens[i]
        formatted = "  ".join(f"{w:.3f}" for w in row)
        LOG.info(f"  {tok!r}: [{formatted}]")

    # Generate a sequence.
    generated = generate_tokens(
        model=model,
        vocab=vocab,
        seed_tokens=seed_tokens,
        num_tokens=num_tokens,
    )
    LOG.info("Generated sequence:")
    LOG.info(f"  {' '.join(generated)}")


if __name__ == "__main__":
    main()
