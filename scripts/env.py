#!/usr/bin/env python3
"""
env.py — 個人設定（.env）を読み込むユーティリティ

優先順位（高い順）:
  1. コマンドライン引数
  2. 環境変数
  3. .env ファイル（sd-lora-factory ルートまたはホームディレクトリ）
  4. 設定ファイル (YAML) の値
  5. デフォルト値

.env の書き方:
  OLLAMA_URL=http://192.168.1.10:11434
"""
import os
from pathlib import Path


def _load_dotenv(path: Path) -> dict:
    """シンプルな .env パーサー（python-dotenv 不要）"""
    result = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def _find_dotenv() -> dict:
    """プロジェクトルートまたはホームディレクトリの .env を探す"""
    # スクリプトの2階層上 = リポジトリルート
    repo_root = Path(__file__).parent.parent
    candidates = [
        repo_root / ".env",
        Path.home() / ".sd-lora-factory.env",
    ]
    for path in candidates:
        env = _load_dotenv(path)
        if env:
            return env
    return {}


_dotenv_cache: dict | None = None


def get_env(key: str, fallback: str = "") -> str:
    """環境変数 → .env → fallback の順で値を返す"""
    global _dotenv_cache
    if _dotenv_cache is None:
        _dotenv_cache = _find_dotenv()

    # 1. 環境変数
    if key in os.environ:
        return os.environ[key]
    # 2. .env ファイル
    if key in _dotenv_cache:
        return _dotenv_cache[key]
    # 3. fallback
    return fallback


def resolve_ollama_url(from_arg: str | None, from_config: str | None) -> str:
    """Ollama URL を優先順位に従って解決する"""
    # 1. コマンドライン引数
    if from_arg:
        return from_arg
    # 2. 環境変数 / .env
    env_url = get_env("OLLAMA_URL")
    if env_url:
        return env_url
    # 3. 設定ファイル
    if from_config:
        return from_config
    # 4. デフォルト
    return "http://127.0.0.1:11434"
