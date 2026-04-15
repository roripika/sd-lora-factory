#!/usr/bin/env python3
"""
build_dataset.py — crawl → vlm_filter → caption を一括実行する

使用例:
    # 全ステップ実行
    python scripts/build_dataset.py --config configs/example.yaml

    # 特定ステップのみ
    python scripts/build_dataset.py --config configs/example.yaml --steps filter caption

    # リモートOllama指定
    python scripts/build_dataset.py --config configs/example.yaml \
        --ollama-url http://100.89.188.24:11434
"""
import argparse
import sys
from pathlib import Path

# スクリプトディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from crawl import crawl
from vlm_filter import vlm_filter
from caption import caption


STEPS = ["crawl", "filter", "caption"]


def main():
    parser = argparse.ArgumentParser(description="LoRA学習データセットを一括構築")
    parser.add_argument("--config", required=True, help="設定ファイルのパス (YAML)")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STEPS,
        default=STEPS,
        help=f"実行するステップ (デフォルト: 全て) 選択肢: {STEPS}",
    )
    parser.add_argument("--ollama-url", default=None, help="Ollama URL (設定ファイルを上書き)")
    parser.add_argument("--max-per-keyword", type=int, default=None, help="クロール: キーワードあたりの最大取得枚数")
    parser.add_argument("--dry-run", action="store_true", help="各ステップを実際には実行せず確認のみ")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  sd-lora-factory: データセット構築パイプライン")
    print(f"  config : {args.config}")
    print(f"  steps  : {args.steps}")
    print(f"  dry-run: {args.dry_run}")
    print(f"{'='*60}\n")

    if "crawl" in args.steps:
        print("[STEP 1/3] crawl —— 画像収集")
        print("-" * 40)
        crawl(args.config, args.max_per_keyword, args.dry_run)
        print()

    if "filter" in args.steps:
        print("[STEP 2/3] vlm_filter —— VLMふるいがけ")
        print("-" * 40)
        vlm_filter(args.config, args.ollama_url, args.dry_run)
        print()

    if "caption" in args.steps:
        print("[STEP 3/3] caption —— キャプション生成")
        print("-" * 40)
        caption(args.config, args.ollama_url, args.dry_run)
        print()

    print("=" * 60)
    print("  完了!")
    print("=" * 60)


if __name__ == "__main__":
    main()
