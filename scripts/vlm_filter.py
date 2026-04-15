#!/usr/bin/env python3
"""
vlm_filter.py — Ollama VLMを使って条件に合う画像をふるいがけする

使用例:
    python scripts/vlm_filter.py --config configs/example.yaml
    python scripts/vlm_filter.py --config configs/example.yaml --ollama-url http://100.89.188.24:11434
"""
import argparse
import base64
import json
import shutil
import sys
from pathlib import Path

import requests
import yaml
from PIL import Image
from tqdm import tqdm

from env import resolve_ollama_url


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    name = cfg["dataset_name"]
    for sec in ["vlm_filter"]:
        for key in ["input_dir", "output_dir"]:
            if key in cfg.get(sec, {}):
                cfg[sec][key] = cfg[sec][key].replace("{dataset_name}", name)
    return cfg


def image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_filter_prompt(must_have: list[str], must_not: list[str]) -> str:
    must_have_str = "\n".join(f"- {item}" for item in must_have)
    must_not_str = "\n".join(f"- {item}" for item in must_not)
    return f"""You are an image filtering assistant for machine learning dataset curation.

Evaluate this image against the following criteria:

MUST HAVE (all must be satisfied):
{must_have_str}

MUST NOT (any violation = reject):
{must_not_str}

Respond ONLY with valid JSON in this exact format:
{{
  "passed": true or false,
  "score": 0.0 to 1.0,
  "must_have_results": [{{"item": "...", "satisfied": true/false, "reason": "..."}}],
  "must_not_results": [{{"item": "...", "violated": true/false, "reason": "..."}}],
  "summary": "brief overall assessment"
}}"""


def evaluate_image(
    image_path: Path,
    must_have: list[str],
    must_not: list[str],
    ollama_url: str,
    model: str,
    min_score: float,
) -> dict:
    prompt = build_filter_prompt(must_have, must_not)
    img_b64 = image_to_base64(image_path)

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "format": "json",
        "options": {"num_ctx": 4096},
    }

    try:
        resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        result = json.loads(raw)
    except (requests.RequestException, json.JSONDecodeError) as e:
        return {"passed": False, "score": 0.0, "error": str(e)}

    # must_not に1つでも違反があれば自動reject
    for check in result.get("must_not_results", []):
        if check.get("violated"):
            result["passed"] = False

    # must_have に1つでも未達があれば自動reject
    for check in result.get("must_have_results", []):
        if not check.get("satisfied"):
            result["passed"] = False

    # スコア閾値チェック
    if result.get("score", 0.0) < min_score:
        result["passed"] = False

    return result


def collect_images(input_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]


def vlm_filter(config_path: str, ollama_url: str | None = None, dry_run: bool = False):
    cfg = load_config(config_path)
    fc = cfg["vlm_filter"]

    input_dir = Path(fc["input_dir"])
    output_dir = Path(fc["output_dir"])
    model = fc.get("model", "qwen2.5vl:7b")
    url = resolve_ollama_url(ollama_url, fc.get("ollama_url"))
    must_have = fc.get("must_have", [])
    must_not = fc.get("must_not", [])
    min_score = fc.get("min_score", 0.7)

    images = collect_images(input_dir)
    print(f"[filter] dataset: {cfg['dataset_name']}")
    print(f"[filter] input  : {input_dir} ({len(images)} 枚)")
    print(f"[filter] output : {output_dir}")
    print(f"[filter] model  : {model} @ {url}")
    print(f"[filter] min_score: {min_score}")

    if dry_run:
        print("[dry-run] VLM評価はスキップします")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    passed = 0
    failed = 0
    errors = 0
    results_log = []

    for img_path in tqdm(images, desc="VLM filter"):
        result = evaluate_image(img_path, must_have, must_not, url, model, min_score)

        log_entry = {"file": str(img_path), **result}
        results_log.append(log_entry)

        if result.get("error"):
            errors += 1
            tqdm.write(f"  [ERROR] {img_path.name}: {result['error']}")
        elif result.get("passed"):
            passed += 1
            dest = output_dir / img_path.name
            shutil.copy2(img_path, dest)
            tqdm.write(f"  [PASS] {img_path.name} score={result.get('score', '?'):.2f}")
        else:
            failed += 1
            tqdm.write(f"  [FAIL] {img_path.name} score={result.get('score', 0.0):.2f} - {result.get('summary', '')[:80]}")

    # 結果ログを保存
    log_path = output_dir / "_filter_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2)

    print(f"\n[filter] 完了: PASS={passed} / FAIL={failed} / ERROR={errors} (合計 {len(images)})")
    print(f"[filter] ログ: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="VLMで画像をふるいがけ")
    parser.add_argument("--config", required=True, help="設定ファイルのパス (YAML)")
    parser.add_argument("--ollama-url", default=None, help="Ollama URL (設定ファイルを上書き)")
    parser.add_argument("--dry-run", action="store_true", help="評価せず設定確認のみ")
    args = parser.parse_args()

    vlm_filter(args.config, args.ollama_url, args.dry_run)


if __name__ == "__main__":
    main()
