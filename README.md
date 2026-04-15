# sd-lora-factory

Stable Diffusion (SDXL) 向け LoRA 学習データセット自動構築パイプライン。

## 特定の小説・プロジェクトに依存しない汎用ツールです。

## 機能

1. **Crawl** — キーワード検索で画像を大量収集
2. **VLM Filter** — Ollama (qwen2.5vl) で条件に合う画像をふるいがけ
3. **Caption** — VLMで学習用キャプションを自動生成
4. **Dataset** — kohya / diffusers-trainer 向けにデータセット整形

## 使い方

```bash
# 環境構築
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# データセット構築（設定ファイルを指定）
python scripts/build_dataset.py --config configs/koita_outfit.yaml
```

## 設定ファイル

`configs/` 以下に YAML で定義します。`configs/example.yaml` を参照。

## ディレクトリ構造

```
sd-lora-factory/
├── scripts/
│   ├── crawl.py           # キーワードクロール
│   ├── vlm_filter.py      # VLMふるいがけ
│   ├── caption.py         # キャプション生成
│   └── build_dataset.py   # 上3つをまとめて実行
├── configs/
│   └── example.yaml       # 設定例
└── datasets/              # .gitignore対象・生成物
    └── {dataset_name}/
        ├── raw/            # ダウンロード生画像
        ├── filtered/       # VLM通過画像
        └── captioned/      # 学習用最終データ（画像+caption.txt）
```

## 依存環境

- Python 3.10+
- Ollama（VLMフィルター・キャプション用）: `qwen2.5vl:7b` 推奨
- icrawler（画像収集）

## Ollama設定

リモートOllamaを使う場合は `--ollama-url` で指定:

```bash
python scripts/build_dataset.py --config configs/koita_outfit.yaml \
    --ollama-url http://100.89.188.24:11434
```
