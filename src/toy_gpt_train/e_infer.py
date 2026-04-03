"""e_infer.py - Inference module (artifact-driven).

Runs inference using previously saved training artifacts.

Responsibilities:
- Load inspectable training artifacts from artifacts/
  - 00_meta.json
  - 01_vocabulary.csv
  - 02_model_weights.csv
- Reconstruct a vocabulary-like interface and model weights
- Generate tokens using greedy decoding (argmax)
- Print top-k next-token probabilities for inspection

Notes:
- This module does NOT retrain by default.
- If artifacts are missing, run d_train.py first.
- Context-3 bootstrapping: generation starts from a single start token. To form the
  first 3-token context, we use (start, start, start) as the initial context.
"""

import argparse
import csv
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import tomllib
from typing import Final

from datafun_toolkit.logger import get_logger, log_header

from toy_gpt_train.c_model import SimpleNextTokenModel
from toy_gpt_train.math_training import argmax
from toy_gpt_train.prompts import parse_args

LOG: logging.Logger = get_logger("INFER", level="INFO")

BASE_DIR: Final[Path] = Path(__file__).resolve().parents[2]
CONFIG_PATH: Final[Path] = BASE_DIR / "config.toml"
ARTIFACTS_DIR: Final[Path] = BASE_DIR / "artifacts"
META_PATH: Final[Path] = ARTIFACTS_DIR / "00_meta.json"
VOCAB_PATH: Final[Path] = ARTIFACTS_DIR / "01_vocabulary.csv"
WEIGHTS_PATH: Final[Path] = ARTIFACTS_DIR / "02_model_weights.csv"

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]


@dataclass(frozen=True)
class ArtifactVocabulary:
    """Vocabulary reconstructed from artifacts/01_vocabulary.csv.

    Provides the same surface area used by inference:
    - vocab_size()
    - get_token_id()
    - get_id_token()
    - get_token_frequency()
    """

    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    token_freq: dict[str, int]

    def vocab_size(self) -> int:
        """Return the total number of tokens in the vocabulary."""
        return len(self.token_to_id)

    def get_token_id(self, token: str) -> int | None:
        """Return the token ID for a given token, or None if not found."""
        return self.token_to_id.get(token)

    def get_id_token(self, idx: int) -> str | None:
        """Return the token for a given token ID, or None if not found."""
        return self.id_to_token.get(idx)

    def get_token_frequency(self, token: str) -> int:
        """Return the frequency count for a given token, or 0 if not found."""
        return self.token_freq.get(token, 0)


def load_config() -> JsonObject:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("rb") as f:
            return tomllib.load(f)
    return {}


def require_artifacts() -> None:
    """Fail fast with a helpful message if artifacts are missing."""
    missing: list[Path] = []
    for p in [META_PATH, VOCAB_PATH, WEIGHTS_PATH]:
        if not p.exists():
            missing.append(p)

    if missing:
        LOG.error("Missing training artifacts:")
        for p in missing:
            LOG.error(f"  - {p}")
        LOG.error("Run training first:")
        LOG.error("  uv run python src/toy_gpt_train/d_train.py")
        raise SystemExit(2)


def load_meta(path: Path) -> JsonObject:
    """Load 00_meta.json."""
    with path.open("r", encoding="utf-8") as f:
        data: JsonObject = json.load(f)
    return data


def load_vocabulary_csv(path: Path) -> ArtifactVocabulary:
    """Load 01_vocabulary.csv -> ArtifactVocabulary."""
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


def load_model_weights_csv(
    path: Path,
    vocab_size: int,
    *,
    expected_rows: int,
) -> list[list[float]]:
    """Load 02_model_weights.csv -> weights matrix."""
    weights: list[list[float]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError("Weights CSV is empty.")
        if len(header) < 2 or header[0] != "input_token":
            raise ValueError("Weights CSV must start with header 'input_token'.")

        num_outputs = len(header) - 1
        if num_outputs != vocab_size:
            raise ValueError(
                f"Weights CSV output width mismatch. Expected {vocab_size} output columns "
                f"but found {num_outputs}."
            )

        for row in reader:
            if not row:
                continue
            if len(row) != vocab_size + 1:
                raise ValueError(
                    f"Invalid weights row length. Expected {vocab_size + 1} columns but found {len(row)}."
                )
            weights.append([float(x) for x in row[1:]])

    if len(weights) != expected_rows:
        raise ValueError(
            f"Weights CSV row count mismatch. Expected {expected_rows} rows but found {len(weights)}."
        )

    return weights


def top_k(probs: list[float], k: int) -> list[tuple[int, float]]:
    """Return top-k (token_id, probability) pairs sorted by probability."""
    pairs: list[tuple[int, float]] = list(enumerate(probs))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:k]


def generate_tokens_context3(
    model: SimpleNextTokenModel,
    vocab: ArtifactVocabulary,
    seed_0: str,
    seed_1: str,
    seed_2: str,
    num_tokens: int,
) -> list[str]:
    """Generate tokens using a context-3 window (t-2, t-1, t).

    Bootstrapping:
        If we only have one start token, we begin with:
            (start, start, start)
        so that forward(previous2_id, previous1_id, current_id) is well-defined.
    """
    generated: list[str] = [seed_0, seed_1, seed_2]

    id_0 = vocab.get_token_id(seed_0)
    id_1 = vocab.get_token_id(seed_1)
    id_2 = vocab.get_token_id(seed_2)

    if id_0 is None or id_1 is None or id_2 is None:
        LOG.error("One or more seed tokens not in vocabulary.")
        return generated

    previous2_id: int = id_0
    previous1_id: int = id_1
    current_id: int = id_2

    for _ in range(num_tokens):
        probs: list[float] = model.forward(previous2_id, previous1_id, current_id)
        next_id: int = argmax(probs)
        next_token: str | None = vocab.get_id_token(next_id)

        if next_token is None:
            LOG.error(f"Generated invalid token ID: {next_id}")
            break

        generated.append(next_token)
        previous2_id = previous1_id
        previous1_id = current_id
        current_id = next_id

    return generated


def main() -> None:
    """Run inference using saved training artifacts."""
    log_header(LOG, "Inference Demo: Load Artifacts and Generate Text (Context-3)")

    require_artifacts()

    meta = load_meta(META_PATH)
    vocab = load_vocabulary_csv(VOCAB_PATH)

    v: int = vocab.vocab_size()
    model = SimpleNextTokenModel(vocab_size=v)
    model.weights = load_model_weights_csv(
        WEIGHTS_PATH,
        vocab_size=v,
        expected_rows=v * v * v,
    )

    args: argparse.Namespace = parse_args([])

    config: JsonObject = load_config()
    infer_config: JsonObject = (
        config.get("infer", {})  # type: ignore[assignment]
        if isinstance(config.get("infer"), dict)
        else {}
    )

    num_tokens: int = args.num_tokens or int(infer_config.get("num_tokens", 10))  # type: ignore[arg-type]
    topk: int = args.topk or int(infer_config.get("topk", 5))  # type: ignore[arg-type]

    # Read 3-token seed from config.
    # All three must exist in the trained vocabulary.
    # Using a sequence that appeared in training produces meaningful predictions.
    seed_0: str = str(infer_config.get("seed_0", ""))
    seed_1: str = str(infer_config.get("seed_1", ""))
    seed_2: str = str(infer_config.get("seed_2", ""))

    # Fall back to most common token repeated if seed is missing.
    if not seed_0 or not seed_1 or not seed_2:
        most_common_token: str = (
            max(vocab.token_freq, key=lambda t: vocab.token_freq[t])
            if vocab.token_freq
            else "<no_tokens>"
        )
        LOG.warning(
            "Seed tokens missing or incomplete in config.toml. "
            f"Falling back to ({most_common_token!r}, {most_common_token!r}, {most_common_token!r}). "
            "Predictions may be uniform for unseen contexts."
        )
        seed_0 = seed_0 or most_common_token
        seed_1 = seed_1 or most_common_token
        seed_2 = seed_2 or most_common_token

    LOG.info(
        f"Loaded repo_name={meta.get('repo_name')} model_kind={meta.get('model_kind')}"
    )
    LOG.info(f"Vocab size: {v}")
    LOG.info(f"Seed context: {seed_0!r}|{seed_1!r}|{seed_2!r}")

    seed_0_id = vocab.get_token_id(seed_0)
    seed_1_id = vocab.get_token_id(seed_1)
    seed_2_id = vocab.get_token_id(seed_2)

    if seed_0_id is not None and seed_1_id is not None and seed_2_id is not None:
        probs: list[float] = model.forward(seed_0_id, seed_1_id, seed_2_id)
        LOG.info(f"Top next-token predictions after {seed_0!r}|{seed_1!r}|{seed_2!r}:")
        for tok_id, prob in top_k(probs, k=max(1, topk)):
            tok = vocab.get_id_token(tok_id)
            LOG.info(f"  {tok!r} (ID {tok_id}): {prob:.4f}")

    generated = generate_tokens_context3(
        model=model,
        vocab=vocab,
        seed_0=seed_0,
        seed_1=seed_1,
        seed_2=seed_2,
        num_tokens=max(0, num_tokens),
    )

    LOG.info("Generated sequence:")
    LOG.info(f"  {' '.join(generated)}")


if __name__ == "__main__":
    main()
