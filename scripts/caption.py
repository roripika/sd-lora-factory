#!/usr/bin/env python3
"""
caption.py — VLMを使って学習用キャプションを自動生成する

使用例:
    python scripts/caption.py --config configs/example.yaml
    python scripts/caption.py --config configs/example.yaml --ollama-url http://100.89.188.24:11434
"""
import argparse
import base64
import json
import shutil
import sys
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

from env import resolve_ollama_url


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    name = cfg["dataset_name"]
    for sec in ["caption"]:
        for key in ["input_dir", "output_dir"]:
            if key in cfg.get(sec, {}):
                cfg[sec][key] = cfg[sec][key].replace("{dataset_name}", name)
    return cfg


def image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_caption(
    image_path: Path,
    prompt: str,
    trigger_word: str,
    ollama_url: str,
    model: str,
    num_ctx: int = 4096,
) -> str:
    img_b64 = image_to_base64(image_path)
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"num_ctx": num_ctx},
    }
    resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    caption = resp.json().get("response", "").strip()

    # キャプションのクリーニング（余分な説明文を除去）
    # 最初の改行以降を削除し、カンマ区切りタグ形式に統一
    caption = caption.split("\n")[0].strip()

    # トリガーワードを先頭に追加
    if trigger_word and not caption.startswith(trigger_word):
        caption = f"{trigger_word}, {caption}"

    return caption


def collect_images(input_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]


def caption(config_path: str, ollama_url: str | None = None, dry_run: bool = False):
    cfg = load_config(config_path)
    cc = cfg["caption"]

    input_dir = Path(cc["input_dir"])
    output_dir = Path(cc["output_dir"])
    model = cc.get("model", "qwen2.5vl:7b")
    url = resolve_ollama_url(ollama_url, cc.get("ollama_url"))
    trigger_word = cc.get("trigger_word", "")
    prompt = cc.get("prompt", "Describe this image with comma-separated tags for Stable Diffusion training.")
    num_ctx = cc.get("num_ctx", 4096)

    images = collect_images(input_dir)
    print(f"[caption] dataset     : {cfg['dataset_name']}")
    print(f"[caption] input       : {input_dir} ({len(images)} 枚)")
    print(f"[caption] output      : {output_dir}")
    print(f"[caption] model       : {model} @ {url}")
    print(f"[caption] trigger_word: {trigger_word or '(なし)'}")

    if dry_run:
        print("[dry-run] キャプション生成はスキップします")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    errors = 0

    for img_path in tqdm(images, desc="Caption"):
        try:
            cap = generate_caption(img_path, prompt, trigger_word, url, model, num_ctx)
        except Exception as e:
            errors += 1
            tqdm.write(f"  [ERROR] {img_path.name}: {e}")
            continue

        # 画像をコピー
        dest_img = output_dir / img_path.name
        shutil.copy2(img_path, dest_img)

        # キャプションをテキストファイルとして保存（kohya形式）
        caption_path = output_dir / (img_path.stem + ".txt")
        caption_path.write_text(cap, encoding="utf-8")

        success += 1
        tqdm.write(f"  [OK] {img_path.name} -> {cap[:80]}...")

    print(f"\n[caption] 完了: 成功={success} / エラー={errors}")
    print(f"[caption] 出力: {output_dir}")
    print(f"[caption] kohya/diffusers-trainer 向けデータセット準備完了")


def main():
    parser = argparse.ArgumentParser(description="VLMで学習用キャプションを自動生成")
    parser.add_argument("--config", required=True, help="設定ファイルのパス (YAML)")
    parser.add_argument("--ollama-url", default=None, help="Ollama URL (設定ファイルを上書き)")
    parser.add_argument("--dry-run", action="store_true", help="生成せず設定確認のみ")
    args = parser.parse_args()

    caption(args.config, args.ollama_url, args.dry_run)


if __name__ == "__main__":
    main()
