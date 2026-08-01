"""ยูทิลิตี้ที่ใช้ร่วมกันทุกสคริปต์ — config, logging, tokenizer, VRAM report."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def get_logger(name: str = "cpt", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    # Windows console เริ่มต้นเป็น cp874 ซึ่ง encode ภาษาไทย + ลูกศรไม่ได้ → logging พัง
    # บังคับ UTF-8 และตกลงมาเป็น errors="replace" ถ้า reconfigure ไม่ได้
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover
        pass
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


LOG = get_logger()


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """โหลด YAML แล้วเข้าถึงแบบ dot-notation ได้ (cfg.model.name)"""

    _data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            value = self._data[key]
        except KeyError as exc:
            raise AttributeError(f"ไม่มี key '{key}' ใน config") from exc
        return Config(value) if isinstance(value, dict) else value

    def get(self, key: str, default: Any = None) -> Any:
        value = self._data.get(key, default)
        return Config(value) if isinstance(value, dict) else value

    def to_dict(self) -> dict[str, Any]:
        return self._data

    def __repr__(self) -> str:
        return f"Config({json.dumps(self._data, ensure_ascii=False)[:120]}...)"


def resolve(path: str | Path) -> Path:
    """แปลง path ใน config ให้เป็น absolute เทียบกับ project root"""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿­"), None)


def normalize_thai(text: str) -> str:
    """
    Normalize ข้อความไทยก่อน tokenize

    ทำไมต้องทำ: สระ/วรรณยุกต์ไทยมีหลาย codepoint sequence ที่ render เหมือนกัน
    ถ้าไม่ normalize โมเดลจะเห็นเป็นคนละ token → เสีย capacity ไปกับความต่างที่ไม่มีความหมาย
    NFC คือรูปแบบที่ tokenizer ของ typhoon-7b ถูกเทรนมา
    """
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_ZERO_WIDTH)
    # ยุบช่องว่างซ้ำ แต่คงย่อหน้าไว้ (ย่อหน้าคือสัญญาณโครงสร้างที่โมเดลควรเรียน)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def thai_char_ratio(text: str) -> float:
    """สัดส่วนอักขระไทย — ใช้กรองข้อความที่หลุดเป็นภาษาอังกฤษล้วน"""
    if not text:
        return 0.0
    thai = sum(1 for ch in text if "฀" <= ch <= "๿")
    return thai / len(text)


# --------------------------------------------------------------------------- #
# model / tokenizer
# --------------------------------------------------------------------------- #
def load_tokenizer(model_name: str, cache_dir: str | None = None):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
    # typhoon-7b (Mistral) ไม่มี pad_token — ต้องตั้งเอง
    # ใช้ eos เป็น pad ได้เพราะเรา mask ด้วย attention_mask อยู่แล้ว
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def vram_report(tag: str = "") -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "cuda: N/A"
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"[VRAM{' ' + tag if tag else ''}] alloc={alloc:.2f}GB peak_reserved={reserved:.2f}GB total={total:.1f}GB"
    except Exception as exc:  # pragma: no cover
        return f"vram_report failed: {exc}"


def count_trainable(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# --------------------------------------------------------------------------- #
# jsonl
# --------------------------------------------------------------------------- #
def read_jsonl(path: str | Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
