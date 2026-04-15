#!/usr/bin/env python3
"""
crawl.py — キーワード検索で画像を大量収集する

使用例:
    python scripts/crawl.py --config configs/example.yaml
    python scripts/crawl.py --config configs/example.yaml --max-per-keyword 50
"""
import argparse
import os
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # {dataset_name} プレースホルダを展開
    name = cfg["dataset_name"]
    for key in ["output_dir"]:
        if key in cfg.get("crawl", {}):
            cfg["crawl"][key] = cfg["crawl"][key].replace("{dataset_name}", name)
    return cfg


def crawl(config_path: str, max_per_keyword: int | None = None, dry_run: bool = False):
    try:
        from icrawler.builtin import BingImageCrawler, GoogleImageCrawler, BaiduImageCrawler
    except ImportError:
        print("[ERROR] icrawler がインストールされていません: pip install icrawler")
        sys.exit(1)

    cfg = load_config(config_path)
    crawl_cfg = cfg["crawl"]
    output_dir = Path(crawl_cfg["output_dir"])
    keywords = crawl_cfg["keywords"]
    engines = crawl_cfg.get("engines", ["bing"])
    limit = max_per_keyword or crawl_cfg.get("max_per_keyword", 100)

    crawler_map = {
        "bing": BingImageCrawler,
        "google": GoogleImageCrawler,
        "baidu": BaiduImageCrawler,
    }

    print(f"[crawl] dataset: {cfg['dataset_name']}")
    print(f"[crawl] output : {output_dir}")
    print(f"[crawl] engines: {engines}")
    print(f"[crawl] limit  : {limit} per keyword")
    print(f"[crawl] keywords ({len(keywords)}):")
    for kw in keywords:
        print(f"  - {kw}")

    if dry_run:
        print("[dry-run] 実際のダウンロードはスキップします")
        return

    total = 0
    for engine_name in engines:
        CrawlerClass = crawler_map.get(engine_name)
        if CrawlerClass is None:
            print(f"[WARNING] 未対応のエンジン: {engine_name}")
            continue

        for i, keyword in enumerate(keywords):
            kw_dir = output_dir / engine_name / f"kw{i:03d}"
            kw_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[crawl] [{engine_name}] '{keyword}' -> {kw_dir}")
            crawler = CrawlerClass(storage={"root_dir": str(kw_dir)})
            crawler.crawl(keyword=keyword, max_num=limit)

            downloaded = len(list(kw_dir.glob("*.jpg")) + list(kw_dir.glob("*.png")) + list(kw_dir.glob("*.jpeg")))
            print(f"[crawl]   -> {downloaded} 枚取得")
            total += downloaded

    print(f"\n[crawl] 完了: 合計 {total} 枚")


def main():
    parser = argparse.ArgumentParser(description="キーワード検索で画像を大量収集")
    parser.add_argument("--config", required=True, help="設定ファイルのパス (YAML)")
    parser.add_argument("--max-per-keyword", type=int, default=None, help="キーワードあたりの最大取得枚数")
    parser.add_argument("--dry-run", action="store_true", help="ダウンロードせず設定確認のみ")
    args = parser.parse_args()

    crawl(args.config, args.max_per_keyword, args.dry_run)


if __name__ == "__main__":
    main()
