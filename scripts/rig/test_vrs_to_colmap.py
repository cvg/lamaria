# scripts/rig/test_vrs_to_colmap.py
import sys
import numpy as np
from pathlib import Path
from lamaria.rig.config.loaders import load_cfg
from lamaria.rig.keyframing.vrs_to_colmap import VrsToColmap
from lamaria.rig.keyframing.image_extraction import run as run_extraction
from lamaria.rig.keyframing.keyframe_selection import KeyframeSelector
from lamaria.rig.optim.triangulation import run as run_triangulation

def main():
    cfg = load_cfg(cli_overrides=sys.argv[1:])
    seq_builder = VrsToColmap(cfg)

    run_extraction(
        image_stream_folder=cfg.image_stream_path,
        vrs_path=cfg.vrs_file_path,
    )
    recon = seq_builder.create()

    out_dir = cfg.result.output_folder_path / cfg.result.init_model
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_builder.write_reconstruction(recon, out_dir)

    seq_builder.write_full_timestamps(cfg.result.output_folder_path / "timestamps" /f"full.npy")

    timestamps = np.load(cfg.result.output_folder_path / "timestamps" /f"full.npy").tolist()

    kf_selector = KeyframeSelector(reconstruction=recon, timestamps=timestamps, cfg=cfg)
    kf_recon = kf_selector.run_keyframing()
    kf_selector.write_reconstruction(kf_recon, cfg.result.output_folder_path / cfg.result.kf_model)
    kf_selector.copy_images_to_keyframes_dir()
    kf_selector.write_keyframe_timestamps(cfg.result.output_folder_path / "timestamps" / f"keyframe.npy")
    _ = run_triangulation(cfg=cfg)


if __name__ == "__main__":
    main()
