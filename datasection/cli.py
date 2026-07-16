from __future__ import annotations

import argparse
from pathlib import Path

from datasection.exporter import verify_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an AIC V1 canonical export")
    parser.add_argument("export_dir", type=Path)
    args = parser.parse_args()
    manifest = verify_export(args.export_dir)
    print(f"OK {manifest.dataset_id}/{manifest.build_id}: {manifest.scene_count} scenes")


if __name__ == "__main__":
    main()
