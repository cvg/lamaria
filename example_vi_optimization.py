import argparse
import shutil
from pathlib import Path

import pycolmap

from lamaria.config.options import (
    KeyframeSelectorOptions,
    TriangulatorOptions,
    VIOptimizerOptions,
)
from lamaria.config.pipeline import PipelineOptions
from lamaria.pipeline.estimate_to_timed_reconstruction import (
    convert_estimate_into_timed_reconstruction,
)
from lamaria.pipeline.keyframe_selection import KeyframeSelector
from lamaria.pipeline.optim.session import SingleSeqSession
from lamaria.pipeline.optim.vi_optimization import VIOptimizer
from lamaria.pipeline.triangulation import run as triangulate
from lamaria.structs.timed_reconstruction import TimedReconstruction
from lamaria.structs.trajectory import Trajectory
from lamaria.structs.vi_reconstruction import VIReconstruction
from lamaria.utils.aria import (
    extract_images_with_timestamps_from_vrs,
    get_imu_data_from_vrs_file,
    initialize_reconstruction_from_vrs_file,
)


def run_estimate_to_timed_recon(
    vrs: Path,
    images_path: Path,
    estimate: Path,
) -> TimedReconstruction:
    """Function to convert a general input
    estimate file to a TimedReconstruction.
    """
    traj = Trajectory.load_from_file(estimate)
    init_recon = initialize_reconstruction_from_vrs_file(vrs)
    timestamps_to_images = extract_images_with_timestamps_from_vrs(
        vrs, images_path
    )
    timed_recon = convert_estimate_into_timed_reconstruction(
        init_recon, traj, timestamps_to_images
    )
    return timed_recon


def run_keyframe_selection(
    options: KeyframeSelectorOptions,
    input_recon: TimedReconstruction,
    images_path: Path,
    keyframes_path: Path,
) -> TimedReconstruction:
    kf_vi_recon = KeyframeSelector.run(
        options,
        input_recon,
        images_path,
        keyframes_path,
    )
    return kf_vi_recon


def run_triangulation(
    options: TriangulatorOptions,
    reference_model_path: Path,
    keyframes_path: Path,
    triangulation_path: Path,
) -> pycolmap.Reconstruction:
    triangulated_model_path = triangulate(
        options,
        reference_model_path,
        keyframes_path,
        triangulation_path,
    )
    return pycolmap.Reconstruction(triangulated_model_path)


def run_optimization(
    vi_options: VIOptimizerOptions,
    triangulator_options: TriangulatorOptions,
    recon: VIReconstruction,
    database_path: Path,
) -> VIReconstruction:
    session = SingleSeqSession(
        vi_options.imu,
        recon,
    )

    optimized_recon = VIOptimizer.optimize(
        vi_options, triangulator_options, session, database_path
    )

    optim_vi_recon = VIReconstruction(
        reconstruction=optimized_recon,
        timestamps=recon.timestamps,
        imu_measurements=recon.imu_measurements,
    )
    return optim_vi_recon


def run_pipeline(
    options: PipelineOptions,
    vrs: Path,
    output_path: Path,
    estimate: Path,
):
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    recon = None

    # Estimate to Lamaria Reconstruction
    image_path = output_path / "images"
    init_recon_path = output_path / "initial_recon"
    if init_recon_path.exists():
        recon = TimedReconstruction.read(init_recon_path)
    else:
        recon = run_estimate_to_timed_recon(
            vrs,
            output_path / "images",
            estimate,
        )

    # Keyframe Selection
    keyframe_path = output_path / "keyframes"
    keyframed_recon_path = output_path / "keyframed_recon"
    if keyframed_recon_path.exists():
        recon = TimedReconstruction.read(keyframed_recon_path)
    else:
        recon = run_keyframe_selection(
            options.keyframing_options,
            recon,
            image_path,
            keyframe_path,
        )
        recon.write(output_path / "keyframed_recon")

    # Triangulation
    triangulation_path = output_path / "triangulated"
    tri_model_path = triangulation_path / "model"
    database_path = tri_model_path / "database.db"
    if tri_model_path.exists():
        recon = TimedReconstruction.read(tri_model_path)
    else:
        pycolmap_recon = run_triangulation(
            options.triangulator_options,
            keyframed_recon_path,
            keyframe_path,
            triangulation_path,
        )
        recon = TimedReconstruction(
            reconstruction=pycolmap_recon, timestamps=recon.timestamps
        )
        recon.write(tri_model_path)

    # Visual-Inertial Optimization
    optim_model_path = output_path / "optim_recon"
    imu_measurements = None
    if optim_model_path.exists():
        imu_measurements = VIReconstruction.read(
            optim_model_path
        ).imu_measurements
        shutil.rmtree(optim_model_path)
    optim_model_path.mkdir(parents=True, exist_ok=True)
    # Load IMU data
    if imu_measurements is None:
        imu_measurements = get_imu_data_from_vrs_file(
            vrs,
        )
    recon = VIReconstruction(
        reconstruction=recon.reconstruction,
        timestamps=recon.timestamps,
        imu_measurements=imu_measurements,
    )
    recon_optimized = run_optimization(
        options.vi_optimizer_options,
        options.triangulator_options,
        recon,
        database_path,
    )
    recon_optimized.write(output_path / "optim_recon")
    return recon_optimized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="./defaults.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--vrs",
        type=str,
        required=True,
        help="Path to the input VRS file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--estimate",
        type=str,
        required=True,
        help="Path to the input estimate file.",
    )
    args = parser.parse_args()

    options = PipelineOptions()
    options.load(args.config)
    run_pipeline(
        options,
        Path(args.vrs),
        Path(args.output),
        Path(args.estimate),
    )
