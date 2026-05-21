import os
import shutil
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SOURCE_DIR = "data/archive"
DEST_DIR = "data/raw"


def parse_safe_filename(filename):

    parts = filename.split("_")

    tile = parts[5]        # T44QLE
    date_raw = parts[2][:8]  # 20260226

    date_obj = datetime.strptime(date_raw, "%Y%m%d")
    date_str = date_obj.strftime("%d-%b-%Y")

    return tile, date_str


def organize_safe_files():

    source = Path(SOURCE_DIR)
    dest_root = Path(DEST_DIR)

    safe_files = list(source.glob("*.SAFE"))

    print(f"Found {len(safe_files)} SAFE files")

    for safe in safe_files:

        filename = safe.name

        try:

            tile, date = parse_safe_filename(filename)

            dest_dir = dest_root / tile / date
            dest_dir.mkdir(parents=True, exist_ok=True)

            dest_path = dest_dir / filename

            print(f"Moving {filename} -> {dest_dir}")

            shutil.move(str(safe), str(dest_path))

        except Exception as e:

            print(f"Skipping {filename} | Error: {e}")

    print("Organization complete.")


if __name__ == "__main__":

    organize_safe_files()
