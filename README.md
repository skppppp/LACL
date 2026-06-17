# LACL: LLM-Augmented Contrastive Learning for Service & Bundle Recommendation

LACL is a contrastive-learning recommender for the Web-API ecosystem. It jointly
models **mashups**, **services (Web APIs)**, **bundles**, and **users**, and—its
core contribution—uses a **Large Language Model to perform *semantic
augmentation*** of the raw, keyword-style textual metadata before it is encoded
into features for the recommender.

---

## Key idea: LLM-based semantic augmentation

The raw dataset stores each item's text as a bag of sparse keyword tokens
(e.g. `["api", "google", "map", "geocoding", ...]`), which is noisy and hard to
embed well. LACL turns this metadata into clean, coherent natural-language
descriptions with an LLM, then embeds those descriptions:

```
raw (name + keywords + category)
        │   LLM semantic augmentation   (data/generation/*)
        ▼
coherent natural-language description
        │   sentence/embedding encoder
        ▼
semantic embeddings  (data/generation/emb/msb_emb.pkl)
        │   LLM-augmented features
        ▼
contrastive recommender                (model/LACL.py)
```

Augmentation is applied at three granularities, each with its own prompt:

| Level | What the LLM produces | Prompt |
|-------|-----------------------|--------|
| **service** | a concise description of a single Web API | `data/generation/service/service_description_prompt.txt` |
| **bundle**  | a description of the capability of a 2–3 service bundle | `data/generation/bundle/bundle_system_prompt.txt` |
| **mashup**  | a summary of which API types a mashup is likely to invoke | `data/generation/mashup/mashup_system_prompt.txt` |

---

## Repository structure

```
LACL/
├── data/                          # dataset (parallel lists, aligned by index)
│   ├── api_name.json              # 23,518 services: names
│   ├── api_description.json       #   keyword tokens per service
│   ├── api_category.json          #   tags per service
│   ├── mashup_name/description/category.json   # 8,217 mashups
│   ├── mashup_used_api.json       #   services each mashup interacts with
│   ├── used_api_list.json         # 1,647 services usable inside bundles
│   ├── bundle_item_matrix.txt     # 2,914 bundles × 1,647 services (0/1)
│   ├── user_item_matrix.txt       # user–service interactions
│   ├── user_bundle_matrix.txt     # user–bundle interactions
│   └── generation/                # ---- LLM semantic augmentation ----
│       ├── llm_client.py          # reusable GPT-4o client (key from env)
│       ├── generate_descriptions.py  # batch-generate descriptions (3 levels)
│       ├── requirements.txt
│       ├── service|bundle|mashup/ # prompt templates + generated *.jsonl
│       └── emb/msb_emb.pkl        # (mashup_emb, bundle_emb, service_emb)
├── model/
│   └── LACL.py                    # model + training/evaluation entry point
├── tools/                         # datasets, metrics, losses, utilities
└── utils/
```

---

## Installation

```bash
# 1) recommender dependencies
pip install torch torchtext numpy scipy

# 2) LLM augmentation dependencies
pip install -r data/generation/requirements.txt   # openai>=1.0.0
```

Set your OpenAI key as an environment variable (it is **never** hardcoded and is
only ever printed masked):

```bash
export OPENAI_API_KEY="sk-...your key..."
```

---

## Usage

### Stage 1 — LLM semantic augmentation

```bash
cd data/generation

# preview the constructed inputs without spending tokens
python generate_descriptions.py service --dry-run --limit 3

# small real batch to validate output quality
python generate_descriptions.py service --limit 20 --workers 4

# full run (resumable: rerun to continue after an interruption)
python generate_descriptions.py all

# let bundle / mashup reuse the polished service descriptions
python generate_descriptions.py bundle --service-desc-file service/service_descriptions.jsonl
python generate_descriptions.py mashup --service-desc-file service/service_descriptions.jsonl
```

Each item is written to JSONL as
`{"id", "name", "task", "parsed": {...}, "raw": "..."}`.

You can also call the client directly from your own code:

```python
from llm_client import LLMClient
client = LLMClient(model="gpt-4o")
out = client.chat_json(system_prompt, '{"name": "...", "keywords": "...", "category": "..."}')
```

### Stage 2 — Build semantic embeddings

Encode the generated descriptions with your sentence/embedding encoder and save
the three matrices as a pickle tuple at `data/generation/emb/msb_emb.pkl`:

```python
# pseudo-code
mashup_emb, bundle_emb, service_emb = encode(descriptions)   # np.ndarray each
pickle.dump((mashup_emb, bundle_emb, service_emb), open("data/generation/emb/msb_emb.pkl", "wb"))
```

`model/LACL.py` loads exactly this tuple as the LLM-augmented features.

### Stage 3 — Train & evaluate the recommender

```bash
python model/LACL.py --lr 1e-6 --latent 512 --batch 128 --device 0 --tag train_ub
```

Key arguments: `--lr`, `--latent` (embedding dim), `--L2`, `--batch`,
`--device`, `--user_weight`, `--bundle_weight`, `--lt_threshold` (long-tail
threshold), `--tag`. Logs are written under `model/log/`.

---

## A note on the choice of LLM

**The contribution of this work is the *method* of LLM-driven semantic
augmentation — how the LLM is prompted at the service / bundle / mashup levels to
turn sparse keyword metadata into recommendation-ready semantic features — rather
than any single model.** Our reported experiments use **GPT-4**, but in our
tests **other large language models reach comparable results**. The pipeline is
deliberately model-agnostic: swap the model with `--model` (e.g.
`--model gpt-4o`) or point `LLMClient` at any OpenAI-compatible endpoint via the
`OPENAI_BASE_URL` environment variable, without changing any downstream code.
