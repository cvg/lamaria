import argparse
import shutil
from pathlib import Path

import pycolmap

from lamaria.config.options import (
    EstimateToLamariaOptions,
    KeyframeSelectorOptions,
    TriangulatorOptions,
    VIOptimizerOptions,
)
from lamaria.config.pipeline import PipelineOptions
from lamaria.pipeline.keyframing.keyframe_selection import KeyframeSelector
from lamaria.pipeline.keyframing.to_lamaria_reconstruction import (
    EstimateToLamaria,
)
from lamaria.pipeline.lamaria_reconstruction import LamariaReconstruction
from lamaria.pipeline.optim.session import SingleSeqSession
from lamaria.pipeline.optim.triangulation import run as triangulate
from lamaria.pipeline.optim.vi_optimization import VIOptimizer


def run_estimate_to_lamaria(
    options: EstimateToLamariaOptions,
    vrs: Path,
    images: Path,
    estimate: Path,
    colmap_model: Path,
) -> LamariaReconstruction:
    """Function to convert a general input
    estimate file to a LamariaReconstruction.
    """
    if colmap_model.exists():
        lamaria_recon = LamariaReconstruction.read(colmap_model)
        return lamaria_recon

    colmap_model.mkdir(parents=True, exist_ok=True)

    lamaria_recon = EstimateToLamaria.convert(
        options,
        vrs,
        images,
        estimate,
    )
    lamaria_recon.write(colmap_model)

    return lamaria_recon


def run_mps_to_lamaria(
    options: EstimateToLamariaOptions,
    vrs: Path,
    images: Path,
    mps_folder: Path,
    colmap_model: Path,
) -> LamariaReconstruction:
    """Function to convert MPS estimate to a LamariaReconstruction."""
    if colmap_model.exists():
        lamaria_recon = LamariaReconstruction.read(colmap_model)
        return lamaria_recon

    colmap_model.mkdir(parents=True, exist_ok=True)

    lamaria_recon = EstimateToLamaria.convert(
        options,
        vrs,
        images,
        mps_folder,
    )
    lamaria_recon.write(colmap_model)

    return lamaria_recon


def run_keyframe_selection(
    options: KeyframeSelectorOptions,
    input: Path | LamariaReconstruction,
    images: Path,
    keyframes_path: Path,
    kf_model: Path,
) -> LamariaReconstruction:
    if isinstance(input, Path):
        input_recon = LamariaReconstruction.read(input)
    else:
        input_recon = input

    if kf_model.exists():
        kf_lamaria_recon = LamariaReconstruction.read(kf_model)
        return kf_lamaria_recon

    kf_model.mkdir(parents=True, exist_ok=True)

    kf_lamaria_recon = KeyframeSelector.run(
        options,
        input_recon,
        images,
        keyframes_path,
    )

    kf_lamaria_recon.write(kf_model)

    return kf_lamaria_recon


def run_triangulation(
    options: TriangulatorOptions,
    input: Path,  # path to LamariaReconstruction
    keyframes_path: Path,
    hloc: Path,
    pairs_file: Path,
    tri_model_path: Path,
) -> LamariaReconstruction:
    if not isinstance(input, Path):
        raise ValueError("Input must be a Path to the reconstruction")

    assert input.exists(), f"input reconstruction path {input} does not exist"

    if tri_model_path.exists():
        tri_lamaria_recon = LamariaReconstruction.read(tri_model_path)
        return tri_lamaria_recon

    triangulated_model_path = triangulate(
        options,
        input,
        keyframes_path,
        hloc,
        pairs_file,
        tri_model_path,
    )

    input_lamaria_recon = LamariaReconstruction.read(input)
    tri_recon = pycolmap.Reconstruction(triangulated_model_path)

    tri_lamaria_recon = LamariaReconstruction()
    tri_lamaria_recon.reconstruction = tri_recon
    tri_lamaria_recon.timestamps = input_lamaria_recon.timestamps
    tri_lamaria_recon.imu_measurements = input_lamaria_recon.imu_measurements
    tri_lamaria_recon.write(triangulated_model_path)

    return tri_lamaria_recon


def run_optimization(
    vi_options: VIOptimizerOptions,
    triangulator_options: TriangulatorOptions,
    input: Path,  # path to LamariaReconstruction
    optim_model_path: Path,
) -> LamariaReconstruction:
    if not isinstance(input, Path):
        raise ValueError("Input must be a Path to the reconstruction")

    db_path = input / "database.db"
    assert db_path.exists(), (
        f"Database path {db_path} does not exist in input reconstruction"
    )

    if optim_model_path.exists():
        shutil.rmtree(optim_model_path)

    optim_model_path.mkdir(parents=True, exist_ok=True)
    db_dst = optim_model_path / "database.db"
    shutil.copy(db_path, db_dst)

    init_lamaria_recon = LamariaReconstruction.read(input)
    session = SingleSeqSession(
        vi_options.imu,
        init_lamaria_recon,
    )

    optimized_recon = VIOptimizer.optimize(
        vi_options, triangulator_options, session, db_dst
    )

    optim_lamaria_recon = LamariaReconstruction()
    optim_lamaria_recon.reconstruction = optimized_recon
    optim_lamaria_recon.timestamps = init_lamaria_recon.timestamps
    optim_lamaria_recon.imu_measurements = init_lamaria_recon.imu_measurements
    optim_lamaria_recon.write(optim_model_path)


def run_pipeline(
    options: PipelineOptions,
    vrs: Path,
    output_path: Path,
    estimate: Path | None = None,
    mps_folder: Path | None = None,
):
    # Setting output path for entire pipeline
    options.output_path = output_path
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # Estimate to Lamaria Reconstruction
    est_options = options.estimate_to_colmap_options
    if not est_options.mps.use_mps:
        assert estimate is not None and estimate.exists(), (
            "Estimate path must be provided if not using MPS"
        )

        _ = run_estimate_to_lamaria(
            est_options,
            vrs,
            options.images,
            estimate,
            options.colmap_model,
        )
    else:
        assert mps_folder is not None and mps_folder.exists(), (
            "MPS folder path must be provided if using MPS"
        )

        _ = run_mps_to_lamaria(
            est_options,
            vrs,
            options.images,
            mps_folder,
            options.colmap_model,
        )

    # Keyframe Selection
    kf_options = options.keyframing_options
    _ = run_keyframe_selection(
        kf_options,
        options.colmap_model,
        options.images,
        options.keyframes_path,
        options.kf_model,
    )

    # Triangulation
    tri_options = options.triangulator_options
    _ = run_triangulation(
        tri_options,
        options.kf_model,
        options.keyframes_path,
        options.hloc,
        options.pairs_file,
        options.tri_model_path,
    )

    # Visual-Inertial Optimization
    vi_options = options.vi_optimizer_options
    _ = run_optimization(
        vi_options,
        tri_options,
        options.tri_model_path,
        options.optim_model_path,
    )


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
