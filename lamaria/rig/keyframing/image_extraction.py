from __future__ import annotations

from pathlib import Path

from lamaria.rig.config.loaders import load_cfg
from ...utils.general import extract_images_from_vrs


def run(
    image_stream_folder: Path,
    vrs_path: Path,
) -> None:

    image_stream_folder.mkdir(parents=True, exist_ok=True)
    extract_images_from_vrs(
        vrs_file=vrs_path,
        image_folder=image_stream_folder,
    )


def main() -> None:
    cfg = load_cfg()
    run(
        image_stream_folder=cfg.image_stream_path,
        vrs_path=cfg.vrs_file_path,
    )


if __name__ == "__main__":
    main()
