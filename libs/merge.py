#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

BASE_DIR = Path("Problem_solving/PhyChem/logs/solve_phy_gpt-3.5-turbo")

def merge_jsonl_in_dir(subdir_name: str, out_name: str = None):

    dir_path = BASE_DIR / subdir_name
    if out_name is None:
        out_name = f"{subdir_name}.jsonl"
    out_path = dir_path / out_name

    if not dir_path.exists():
        print(f"[WARN] Directory not found, skip: {dir_path}")
        return

    jsonl_files = sorted(
        f for f in dir_path.glob("*.jsonl")
        if f.name != out_name  # avoid reading the merged file itself
    )

    if not jsonl_files:
        print(f"[INFO] No .jsonl files found under {dir_path}")
        return

    print(f"[INFO] {subdir_name}: found {len(jsonl_files)} jsonl files:")
    # for f in jsonl_files:
    #     print("  -", f.name)

    with out_path.open("w", encoding="utf-8") as fout:
        for f in jsonl_files:
            with f.open("r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)

    print(f"[DONE] Merged {len(jsonl_files)} files into: {out_path}\n")


def main():
    # correct -> correct.jsonl
    merge_jsonl_in_dir("correct", out_name="correct.jsonl")
    # sol -> sol.jsonl
    merge_jsonl_in_dir("sol", out_name="sol.jsonl")
    # wrong -> wrong.jsonl
    merge_jsonl_in_dir("wrong", out_name="wrong.jsonl")


if __name__ == "__main__":
    main()
