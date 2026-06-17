#!/usr/bin/env python3
"""Generate LLM descriptions for services / bundles / mashups in the LACL dataset.

For each of the three tasks it:
  1. reads name / description(keywords) / category from ``LACL/data/*.json``
  2. fills the matching prompt template under
     ``data/generation/{service,bundle,mashup}/*.txt`` as the system prompt
  3. calls the OpenAI Chat Completions API once per item
  4. appends the result to a JSONL file next to the prompt

Data layout (all parallel lists, aligned by index):
  service : api_name.json / api_description.json / api_category.json   (23518)
  bundle  : bundle_item_matrix.txt -> used_api_list.json -> services   (2914)
  mashup  : mashup_name/description/category.json + mashup_used_api     (8217)

The API key is read from the environment variable OPENAI_API_KEY.
NEVER hardcode the key here, and never commit a .env file.

Examples:
  export OPENAI_API_KEY=sk-...
  # preview the constructed inputs without spending any tokens:
  python generate_descriptions.py service --dry-run --limit 3
  # run a small real batch:
  python generate_descriptions.py service --limit 20 --workers 4
  # run everything (resumable; rerun to continue after an interruption):
  python generate_descriptions.py all
  # let bundle/mashup reuse the polished service descriptions:
  python generate_descriptions.py bundle \
      --service-desc-file service/service_descriptions.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_client import LLMClient, extract_json

HERE = Path(__file__).resolve().parent          # .../data/generation
DATA = HERE.parent                              # .../data

PROMPTS = {
    "service": HERE / "service" / "service_description_prompt.txt",
    "bundle": HERE / "bundle" / "bundle_system_prompt.txt",
    "mashup": HERE / "mashup" / "mashup_system_prompt.txt",
}
OUTPUTS = {
    "service": HERE / "service" / "service_descriptions.jsonl",
    "bundle": HERE / "bundle" / "bundle_descriptions.jsonl",
    "mashup": HERE / "mashup" / "mashup_descriptions.jsonl",
}


# --------------------------------------------------------------------------- #
# data helpers
# --------------------------------------------------------------------------- #
def load_json(name: str):
    with open(DATA / name, encoding="utf-8") as f:
        return json.load(f)


def join_tokens(x) -> str:
    """description fields are stored as token lists -> a readable sentence."""
    if isinstance(x, list):
        return " ".join(map(str, x))
    return str(x or "")


def join_tags(x) -> str:
    if isinstance(x, list):
        return ", ".join(map(str, x))
    return str(x or "")


def make_text_desc_lookup(service_desc_file: str | None):
    """Return f(service_idx, descs) -> text_description.

    If a previously generated service description file is supplied, prefer the
    polished LLM description; otherwise fall back to the raw keyword tokens.
    """
    generated: dict[int, str] = {}
    if service_desc_file:
        p = Path(service_desc_file)
        if not p.is_absolute():
            p = HERE / p
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        rec = json.loads(ln)
                        parsed = rec.get("parsed") or {}
                        desc = parsed.get("description")
                        if desc and rec.get("id") is not None:
                            generated[int(rec["id"])] = desc
                    except Exception:
                        continue
            print(f"[lookup] loaded {len(generated)} generated service descriptions")

    def lookup(idx: int, descs) -> str:
        if idx in generated:
            return generated[idx]
        return join_tokens(descs[idx])

    return lookup


# --------------------------------------------------------------------------- #
# input builders  (each item -> {id, name, user_content, [services]})
# --------------------------------------------------------------------------- #
def build_service_inputs():
    names = load_json("api_name.json")
    descs = load_json("api_description.json")
    cats = load_json("api_category.json")
    items = []
    for i, name in enumerate(names):
        payload = {
            "name": name,
            "keywords": join_tokens(descs[i]),
            "category": join_tags(cats[i]),
        }
        items.append(
            {"id": i, "name": name,
             "user_content": json.dumps(payload, ensure_ascii=False)}
        )
    return items


def build_bundle_inputs(lookup):
    used = load_json("used_api_list.json")          # column index -> api name
    names = load_json("api_name.json")
    descs = load_json("api_description.json")
    cats = load_json("api_category.json")
    name2idx = {n: i for i, n in enumerate(names)}

    items = []
    with open(DATA / "bundle_item_matrix.txt", encoding="utf-8") as f:
        for b, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            vals = line.split(",")
            svc_names = [used[j] for j, v in enumerate(vals) if v == "1" and j < len(used)]
            svc_list = []
            for sn in svc_names:
                idx = name2idx.get(sn)
                if idx is None:
                    continue
                svc_list.append({
                    "name": sn,
                    "text_description": lookup(idx, descs),
                    "category": join_tags(cats[idx]),
                })
            items.append({
                "id": b,
                "name": f"bundle_{b}",
                "services": svc_names,
                "user_content": json.dumps(svc_list, ensure_ascii=False),
            })
    return items


def build_mashup_inputs(lookup):
    mnames = load_json("mashup_name.json")
    mdescs = load_json("mashup_description.json")
    mcats = load_json("mashup_category.json")
    mused = load_json("mashup_used_api.json")
    names = load_json("api_name.json")
    descs = load_json("api_description.json")
    cats = load_json("api_category.json")
    name2idx = {n: i for i, n in enumerate(names)}

    items = []
    for i, mn in enumerate(mnames):
        mashup_info = {
            "name": mn,
            "text_description": join_tokens(mdescs[i]),
            "category": join_tags(mcats[i]),
        }
        interacted = []
        for sn in mused[i]:
            idx = name2idx.get(sn)
            if idx is None:
                continue
            interacted.append({
                "interacted_service": sn,
                "text_description": lookup(idx, descs),
                "category": join_tags(cats[idx]),
            })
        payload = [mashup_info] + interacted
        items.append({
            "id": i,
            "name": mn,
            "used_api": mused[i],
            "user_content": json.dumps(payload, ensure_ascii=False),
        })
    return items


BUILDERS = {
    "service": build_service_inputs,
    "bundle": build_bundle_inputs,
    "mashup": build_mashup_inputs,
}


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def load_done_ids(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    done.add(json.loads(ln)["id"])
                except Exception:
                    continue
    return done


def run_task(task: str, args):
    system = PROMPTS[task].read_text(encoding="utf-8")
    lookup = make_text_desc_lookup(args.service_desc_file)
    items = BUILDERS[task](lookup) if task != "service" else BUILDERS[task]()

    out_path = Path(args.output) if args.output else OUTPUTS[task]
    if not out_path.is_absolute():
        out_path = HERE / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and out_path.exists():
        out_path.unlink()

    # slice for testing
    items = items[args.start:]
    if args.limit:
        items = items[:args.limit]

    # -------- dry run: just dump the constructed inputs, no API call -------- #
    if args.dry_run:
        preview = out_path.with_suffix(".preview.jsonl")
        with open(preview, "w", encoding="utf-8") as f:
            for it in items:
                rec = {"id": it["id"], "name": it["name"], "task": task,
                       "input": json.loads(it["user_content"])}
                if "services" in it:
                    rec["services"] = it["services"]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{task}] DRY RUN wrote {len(items)} previews -> {preview}")
        return

    client = LLMClient(model=args.model)  # reads OPENAI_API_KEY from the environment

    done = load_done_ids(out_path)
    todo = [it for it in items if it["id"] not in done]
    print(f"[{task}] selected={len(items)} already_done={len(done)} "
          f"todo={len(todo)} -> {out_path}")

    lock = threading.Lock()
    counters = {"ok": 0, "fail": 0}
    fh = open(out_path, "a", encoding="utf-8")

    def work(it):
        raw = client.chat(system, it["user_content"])
        parsed = extract_json(raw)
        rec = {"id": it["id"], "name": it["name"], "task": task,
               "parsed": parsed, "raw": raw}
        if "services" in it:
            rec["services"] = it["services"]
        if "used_api" in it:
            rec["used_api"] = it["used_api"]
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
        return parsed is not None

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, it) for it in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    good = fut.result()
                    counters["ok" if good else "fail"] += 1
                except Exception as e:  # noqa: BLE001
                    counters["fail"] += 1
                    sys.stderr.write(f"item failed: {e}\n")
                if i % 20 == 0:
                    print(f"  [{task}] {i}/{len(todo)} "
                          f"ok={counters['ok']} fail={counters['fail']}")
    finally:
        fh.close()
    print(f"[{task}] DONE ok={counters['ok']} parse_fail={counters['fail']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["service", "bundle", "mashup", "all"])
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o"),
                    help="OpenAI model (default: gpt-4o)")
    ap.add_argument("--workers", type=int, default=4, help="concurrent requests")
    ap.add_argument("--limit", type=int, default=0, help="only first N items (0 = all)")
    ap.add_argument("--start", type=int, default=0, help="skip the first N items")
    ap.add_argument("--output", default="", help="override output JSONL path")
    ap.add_argument("--overwrite", action="store_true",
                    help="truncate the output file before running")
    ap.add_argument("--service-desc-file", default="",
                    help="generated service descriptions JSONL; lets bundle/mashup "
                         "reuse polished service descriptions instead of raw keywords")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and dump inputs only, no API calls / no key needed")
    args = ap.parse_args()

    tasks = ["service", "bundle", "mashup"] if args.task == "all" else [args.task]

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=sk-...")

    for t in tasks:
        run_task(t, args)


if __name__ == "__main__":
    main()
