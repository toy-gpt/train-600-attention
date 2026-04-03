# train-600-attention

[![Docs](https://img.shields.io/badge/docs-live-blue)](https://toy-gpt.github.io/train-600-attention/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
[![CI](https://github.com/toy-gpt/train-600-attention/actions/workflows/ci-shared.yml/badge.svg?branch=main)](https://github.com/toy-gpt/train-600-attention/actions/workflows/ci-shared.yml)
[![Deploy-Docs](https://github.com/toy-gpt/train-600-attention/actions/workflows/deploy-docs-shared.yml/badge.svg?branch=main)](https://github.com/toy-gpt/train-600-attention/actions/workflows/deploy-docs-shared.yml)
[![Check Links](https://github.com/toy-gpt/train-600-attention/actions/workflows/links.yml/badge.svg)](https://github.com/toy-gpt/train-600-attention/actions/workflows/links.yml)
[![Dependabot](https://img.shields.io/badge/Dependabot-enabled-brightgreen.svg)](https://github.com/toy-gpt/train-600-attention/security)

> Demonstrates, at very small scale, how a language model learns to dynamically weight context tokens using self-attention.

This repository is part of a series of toy training repositories plus a companion client repository:

- [**Training repositories**](https://github.com/toy-gpt) produce pretrained artifacts (vocabulary, weights, metadata).
- A [**web app**](https://toy-gpt.github.io/toy-gpt-chat/) loads the artifacts and provides an interactive prompt.

## What is different about this model (600 vs 500)

The embeddings model (500) concatenates all context vectors and
passes them through a fixed linear layer.
Every context position is weighted equally.
The model has no way to learn that one preceding token
matters more than another for a given prediction.

This repository introduces **single-head self-attention**:
instead of treating all context positions equally,
the model learns query, key, and value projections that
allow it to dynamically weight which context tokens
matter most for each prediction.

The key mechanism:
for each prediction,
a **query** vector is computed from the current position and
compared against **key** vectors from all context positions.
The similarity scores determine how much
each position's **value** vector contributes to the output.
This is the core idea behind the Transformer architecture.

Reference: Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (2017).

## Concepts

<details>
<summary>Query, Key, Value (Q, K, V)</summary>

Three learned projections applied to each token embedding:

- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What do I contribute?"

For each prediction, the query at the last context position is compared
against all keys via dot product. The resulting scores, after softmax,
become attention weights that determine how much each value contributes
to the context vector.

</details>

<details>
<summary>Scaled dot-product attention</summary>

```
scores[i][j] = dot(Q[i], K[j]) / sqrt(head_dim)
weights[i]   = softmax(scores[i])
output[i]    = sum_j( weights[i][j] * V[j] )
```

The scaling by `sqrt(head_dim)` prevents dot products from growing too large
in higher dimensions, which would push softmax into a near-zero gradient region.

</details>

<details>
<summary>Attention weights</summary>

A probability distribution over context positions,
computed from Q·K scores.
After training, these weights show which context positions the model has
learned to attend to most.
They are displayed by `e_infer.py` at inference time
and are not written to a separate artifact file.
They are recomputed from the loaded model parameters on each run.

</details>

<details>
<summary>Positional embeddings</summary>

A learned vector added to each token embedding **before** Q/K/V projection,
giving the model information about token position in the context window.

Without positional embeddings, attention is **position-invariant**: the model
cannot distinguish "token at position 0" from "token at position 1" and
attention weights never break symmetry from [0.5, 0.5].

This is one of the most important insights in the Transformer architecture.

</details>

<details>
<summary>Gradient clipping</summary>

A training stabilization technique that caps individual gradient values
before applying weight updates. Gradients compound through Q/K/V projections
and the attention softmax; without clipping, a single large gradient can
destroy all learned weights (producing `nan` loss).

In this repo: `MAX_GRAD = 1.0` applied to all parameter updates.

</details>

<details>
<summary>Backpropagation through attention</summary>

Gradients flow through six parameter groups per training example:

1. `W_out` (output projection)
2. `bias`
3. `W_V` (value projection) via the attention weighted sum
4. `W_Q` (query projection) via the attention score softmax
5. `W_K` (key projection) via the attention score softmax
6. `embeddings` and `pos_embeddings` via all three projection paths

Compare to the embeddings model (500) which updates two parameter groups:
the linear weights and the token embeddings.

</details>

<details>
<summary>⚠️ Why attention weights stay uniform on this corpus</summary>

After 100 epochs the attention weights remain [0.500, 0.500].
Both context positions receive equal weight.
This is expected and instructive - it's not wrong.

Three conditions are needed for attention to break symmetry:

1. **Positional signal**: added via `pos_embeddings` in this repo
2. **Sufficient data**: 226 training pairs is near the minimum; attention
   needs many distinct contexts to learn position-specific patterns
3. **Strong Q/K gradient signal**: the gradient through the attention
   softmax is proportional to Q·K magnitude, which starts near zero with
   small random initialization and stays small when `W_out` learning dominates

Real transformers address this with larger datasets,
multiple attention heads, warmup learning rate schedules,
and careful initialization.
This repo shows the mechanism correctly;
the limitation is the scale, not the implementation.

On a larger corpus with more epochs the attention weights
would show meaningful asymmetry.

</details>

## Training observations (this corpus, 100 epochs)

| Epoch | Avg loss | Accuracy |
|------:|--------:|--------:|
| 1     | 4.695   | 0.142   |
| 10    | 4.356   | 0.150   |
| 50    | 4.220   | 0.150   |
| 100   | 4.205   | 0.150   |

Loss declines steadily but accuracy does not improve beyond 0.150.
The model is learning through `W_out` but the attention pathway
does not develop meaningful position weighting at this scale.

Attention weights after training: `['data': [0.500  0.500], 'analytics': [0.500  0.500]]`

## Parameter count vs earlier models

| Model         |   Vocab | Parameters | Notes                                  |
|---------------|--------:|-----------:|----------------------------------------|
| Unigram       |     112 |        112 | one score per token                    |
| Bigram        |     112 |     12,544 | vocab²                                 |
| Context-2     |     112 |  1,404,928 | vocab³                                 |
| Context-3     |     112 | 157,351,936 | vocab⁴, mostly zeros                 |
| Embeddings    |     112 |      5,488 | 1,792 + 3,584 + 112                    |
| **Attention** | **112** |  **4,496** | 1,792 + 32 + 768 + 1,792 + 112        |

The attention model has fewer parameters than the embeddings model
because it replaces the wide concatenation + linear layer with
compact Q/K/V projections plus an output projection.

## Artifacts

Training produces the following files under `artifacts/`:

| File | Contents |
|------|----------|
| `00_meta.json` | Corpus hash, model kind, training settings, concept glossary |
| `01_vocabulary.csv` | token_id, token, frequency |
| `02_model_weights.csv` | W_out weights: head_dim rows × vocab_size columns |
| `03_token_embeddings.csv` | Learned token embedding vectors, one row per vocabulary token |
| `04_positional_embeddings.csv` | Learned positional embedding vectors, one row per context position |

Training logs are written to `outputs/train_log.csv` (epoch, avg_loss, accuracy).

## Contents

- `corpus/` - declared training corpus (`030_analytics.txt`)
- `src/toy_gpt_train/` - tokenizer, vocabulary, model, training loop, inference, I/O utilities
- `artifacts/` - committed pretrained artifacts for downstream use
- `outputs/` - training logs (not committed)

## Scope

This is an educational, inspectable training pipeline,
a next-token predictor trained on an explicit corpus.
It is not a production system, a full Transformer,
a chat interface, or a claim of semantic understanding.

## Quick start

```shell
uv run python src/toy_gpt_train/d_train.py
```

Run individual pipeline steps:

```shell
uv run python src/toy_gpt_train/a_tokenizer.py
uv run python src/toy_gpt_train/b_vocab.py
uv run python src/toy_gpt_train/c_model.py
uv run python src/toy_gpt_train/d_train.py
uv run python src/toy_gpt_train/e_infer.py
```

<details>
<summary>Command reference</summary>

### In a machine terminal (open in your `Repos` folder)

```shell
# Replace username with YOUR GitHub username.
git clone https://github.com/username/train-600-attention
cd train-600-attention
code .
```

### In a VS Code terminal

```shell
uv self update
uv python pin 3.14
uv sync --extra dev --extra docs --upgrade

uvx pre-commit install
git add -A
uvx pre-commit run --all-files

uv run python src/toy_gpt_train/a_tokenizer.py
uv run python src/toy_gpt_train/b_vocab.py
uv run python src/toy_gpt_train/c_model.py
uv run python src/toy_gpt_train/d_train.py
uv run python src/toy_gpt_train/e_infer.py

uv run ruff format .
uv run ruff check . --fix
uv run zensical build

git add -A
git commit -m "update"
git push -u origin main
```

</details>

## Provenance and Purpose

The primary corpus used for training is declared in `SE_MANIFEST.toml`.
This repository commits pretrained artifacts so the client can run without retraining.

## Resources

- [Toy GPT organization](https://github.com/toy-gpt) — all training repositories
- [ANNOTATIONS.md](./ANNOTATIONS.md) — REQ/WHY/OBS annotations used
- [SE_MANIFEST.toml](./SE_MANIFEST.toml) — project intent, scope, and declared corpus

## Citation

[CITATION.cff](./CITATION.cff)

## License
