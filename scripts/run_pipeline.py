import sys
import numpy as np
from lamaria.rig.config.loaders import load_cfg
from lamaria.rig.keyframing.vrs_to_colmap import VrsToColmap
from lamaria.rig.keyframing.image_extraction import run as run_extraction
from lamaria.rig.keyframing.keyframe_selection import KeyframeSelector
from lamaria.rig.optim.triangulation import run as run_triangulation
from lamaria.rig.optim.session import SingleSeqSession
from lamaria.rig.optim.vi_optimization import run as run_vi_optimization



def run_pipeline():
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
    seq_builder.write_full_timestamps(cfg.result.output_folder_path / cfg.result.full_ts)

    timestamps = np.load(cfg.result.output_folder_path / cfg.result.full_ts).tolist()

    kf_selector = KeyframeSelector(reconstruction=recon, timestamps=timestamps, cfg=cfg)
    kf_recon = kf_selector.run_keyframing()
    kf_selector.write_reconstruction(kf_recon, cfg.result.output_folder_path / cfg.result.kf_model)
    kf_selector.copy_images_to_keyframes_dir()
    kf_selector.write_keyframe_timestamps(cfg.result.output_folder_path / cfg.result.kf_ts)
    triangulated_model_path = run_triangulation(cfg=cfg)

    timestamps = kf_selector.read_keyframe_timestamps(cfg.result.output_folder_path / cfg.result.kf_ts)
    rect_imu_file = cfg.result.output_folder_path / cfg.result.rect_imu_file
    if not rect_imu_file.exists():
        raise FileNotFoundError(f"Rectified IMU file not found at {rect_imu_file}")

    session = SingleSeqSession(
        reconstruction=kf_recon,
        timestamps=timestamps,
        rect_imu_file=rect_imu_file,
    )
    optimized_recon = run_vi_optimization(
        session=session,
        database_path=triangulated_model_path / "database.db",
        output_folder=cfg.result.output_folder_path / cfg.result.optim_model,
    )

        


if __name__ == "__main__":
    run_pipeline()
