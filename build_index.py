#!/usr/bin/env python3
"""
BookLens AI Build Script
Bundles assets_data.json into template.html to produce index.html.
"""

import json
import os

def build():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_path = os.path.join(base_dir, "assets_data.json")
    template_path = os.path.join(base_dir, "template.html")
    output_path = os.path.join(base_dir, "index.html")

    if not os.path.exists(assets_path) or not os.path.exists(template_path):
        print(f"Error: Required files not found in {base_dir}")
        return

    with open(assets_path, "r", encoding="utf-8") as f:
        assets = json.load(f)

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    final_html = template.replace("/* __ASSET_DATA_JSON__ */", json.dumps(assets))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    print(f"Built index.html successfully! Size: {len(final_html):,} bytes")

if __name__ == "__main__":
    build()
