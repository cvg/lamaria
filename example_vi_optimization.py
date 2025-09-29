import argparse
import shutil
from pathlib import Path

import pycolmap

from lamaria.config.options import (
    EstimateToTimedReconOptions,
    KeyframeSelectorOptions,
    TriangulatorOptions,
    VIOptimizerOptions,
)
from lamaria.config.pipeline import PipelineOptions
from lamaria.pipeline.keyframing.keyframe_selection import KeyframeSelector
from lamaria.pipeline.keyframing.to_vi_reconstruction import (
    EstimateToTimedRecon,
)
from lamaria.pipeline.optim.session import SingleSeqSession
from lamaria.pipeline.optim.triangulation import run as triangulate
from lamaria.pipeline.optim.vi_optimization import VIOptimizer
from lamaria.structs.timed_reconstruction import TimedReconstruction
from lamaria.structs.vi_reconstruction import VIReconstruction
from lamaria.utils.aria import get_imu_data_from_vrs_file


def run_estimate_to_vi_recon(
    options: EstimateToTimedReconOptions,
    vrs: Path,
    images_path: Path,
    estimate: Path,
) -> TimedReconstruction:
    """Function to convert a general input
    estimate file to a TimedReconstruction.
    """
    vi_recon = EstimateToTimedRecon.convert(
        options,
        vrs,
        images_path,
        estimate,
    )
    return vi_recon


def run_mps_to_vi_recon(
    options: EstimateToTimedReconOptions,
    vrs: Path,
    images_path: Path,
    mps_folder: Path,
) -> TimedReconstruction:
    """Function to convert MPS estimate to a TimedReconstruction."""
    vi_recon = EstimateToTimedRecon.convert(
        options,
        vrs,
        images_path,
        mps_folder,
    )
    return vi_recon


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

    optim_vi_recon = VIReconstruction(reconstruction = optimized_recon, timestamps = recon.timestamps, imu_measurements = recon.imu_measurements)
    return optim_vi_recon


def run_pipeline(
    options: PipelineOptions,
    vrs: Path,
    output_path: Path,
    estimate: Path | None = None,
    mps_folder: Path | None = None,
):
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    recon = None


    # Estimate to Lamaria Reconstruction
    src = "estimate" if estimate is not None else "mps"
    image_path = output_path / "images"
    init_recon_path = output_path / "initial_recon"
    if init_recon_path.exists():
        recon = TimedReconstruction.read(init_recon_path)
    else:
        if src == "estimate":
            assert estimate.exists(), (
                "Estimate path must be provided if not using MPS"
            )

            recon = run_estimate_to_vi_recon(
                options.estimate_to_colmap_options,
                vrs,
                output_path / "images",
                estimate,
            )

        else:
            assert mps_folder.exists(), (
                "MPS folder path must be provided if using MPS"
            )

            recon = run_mps_to_vi_recon(
                options.estimate_to_colmap_options,
                vrs,
                output_path / "images",
                mps_folder,
            )
            recon.write(init_recon_path)

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
        recon = TimedReconstruction(reconstruction=pycolmap_recon, timestamps=recon.timestamps)
        recon.write(tri_model_path)

    # Visual-Inertial Optimization
    optim_model_path = output_path / "optim_recon"
    imu_measurements = None
    if optim_model_path.exists():
        imu_measurements = VIReconstruction.read(optim_model_path).imu_measurements
        shutil.rmtree(optim_model_path)
    optim_model_path.mkdir(parents=True, exist_ok=True)
    # Load IMU data
    if imu_measurements is None:
        if options.vi_optimizer_options.use_mps_online_calibration:
            assert mps_folder is not None, (
                "MPS folder path must be provided if using MPS"
            )
        imu_measurements = get_imu_data_from_vrs_file(
            vrs, mps_folder if options.vi_optimizer_options.use_mps_online_calibration else None
        )
    recon = VIReconstruction(reconstruction=recon.reconstruction, timestamps=recon.timestamps, imu_measurements=imu_measurements)
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
        default=None,
        help="Path to the input estimate file (if not using MPS).",
    )
    parser.add_argument(
        "--mps_folder",
        type=str,
        default=None,
        help="Path to the input MPS folder (if using MPS).",
    )
    args = parser.parse_args()

    # ensure either estimate or mps_folder is provided
    if args.estimate is None and args.mps_folder is None:
        parser.error("Either --estimate or --mps_folder must be provided.")
    if args.estimate is not None and args.mps_folder is not None:
        parser.error(
            "Only one of --estimate or --mps_folder should be provided."
        )

    options = PipelineOptions()
    options.load(args.config)
    run_pipeline(
        options,
        Path(args.vrs),
        Path(args.output),
        estimate=Path(args.estimate) if args.estimate else None,
        mps_folder=Path(args.mps_folder) if args.mps_folder else None,
    )
