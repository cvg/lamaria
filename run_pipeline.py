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
from lamaria.pipeline.keyframing.to_vi_reconstruction import (
    EstimateToLamaria,
)
from lamaria.pipeline.optim.session import SingleSeqSession
from lamaria.pipeline.optim.triangulation import run as triangulate
from lamaria.pipeline.optim.vi_optimization import VIOptimizer
from lamaria.structs.vi_reconstruction import VIReconstruction


def run_estimate_to_lamaria(
    options: EstimateToLamariaOptions,
    vrs: Path,
    images_path: Path,
    estimate: Path,
    colmap_model_path: Path,
) -> VIReconstruction:
    """Function to convert a general input
    estimate file to a VIReconstruction.
    """
    if colmap_model_path.exists():
        vi_recon = VIReconstruction.read(colmap_model_path)
        return vi_recon

    colmap_model_path.mkdir(parents=True, exist_ok=True)

    vi_recon = EstimateToLamaria.convert(
        options,
        vrs,
        images_path,
        estimate,
    )
    vi_recon.write(colmap_model_path)

    return vi_recon


def run_mps_to_lamaria(
    options: EstimateToLamariaOptions,
    vrs: Path,
    images_path: Path,
    mps_folder: Path,
    colmap_model_path: Path,
) -> VIReconstruction:
    """Function to convert MPS estimate to a VIReconstruction."""
    if colmap_model_path.exists():
        vi_recon = VIReconstruction.read(colmap_model_path)
        return vi_recon

    colmap_model_path.mkdir(parents=True, exist_ok=True)

    vi_recon = EstimateToLamaria.convert(
        options,
        vrs,
        images_path,
        mps_folder,
    )
    vi_recon.write(colmap_model_path)

    return vi_recon


def run_keyframe_selection(
    options: KeyframeSelectorOptions,
    input: Path | VIReconstruction,
    images_path: Path,
    keyframes_path: Path,
    kf_model_path: Path,
) -> VIReconstruction:
    if isinstance(input, Path):
        input_recon = VIReconstruction.read(input)
    else:
        input_recon = input

    if kf_model_path.exists():
        kf_vi_recon = VIReconstruction.read(kf_model_path)
        return kf_vi_recon

    kf_model_path.mkdir(parents=True, exist_ok=True)

    kf_vi_recon = KeyframeSelector.run(
        options,
        input_recon,
        images_path,
        keyframes_path,
    )

    kf_vi_recon.write(kf_model_path)

    return kf_vi_recon


def run_triangulation(
    options: TriangulatorOptions,
    input: Path,  # path to VIReconstruction
    keyframes_path: Path,
    hloc_path: Path,
    pairs_file: Path,
    tri_model_path: Path,
) -> VIReconstruction:
    if not isinstance(input, Path):
        raise ValueError("Input must be a Path to the reconstruction")

    assert input.exists(), f"input reconstruction path {input} does not exist"

    if tri_model_path.exists():
        tri_vi_recon = VIReconstruction.read(tri_model_path)
        return tri_vi_recon

    triangulated_model_path = triangulate(
        options,
        input,
        keyframes_path,
        hloc_path,
        pairs_file,
        tri_model_path,
    )

    input_vi_recon = VIReconstruction.read(input)
    tri_recon = pycolmap.Reconstruction(triangulated_model_path)

    tri_vi_recon = VIReconstruction()
    tri_vi_recon.reconstruction = tri_recon
    tri_vi_recon.timestamps = input_vi_recon.timestamps
    tri_vi_recon.imu_measurements = input_vi_recon.imu_measurements
    tri_vi_recon.write(triangulated_model_path)

    return tri_vi_recon


def run_optimization(
    vi_options: VIOptimizerOptions,
    triangulator_options: TriangulatorOptions,
    input: Path,  # path to VIReconstruction
    optim_model_path: Path,
) -> VIReconstruction:
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

    init_vi_recon = VIReconstruction.read(input)
    session = SingleSeqSession(
        vi_options.imu,
        init_vi_recon,
    )

    optimized_recon = VIOptimizer.optimize(
        vi_options, triangulator_options, session, db_dst
    )

    optim_vi_recon = VIReconstruction()
    optim_vi_recon.reconstruction = optimized_recon
    optim_vi_recon.timestamps = init_vi_recon.timestamps
    optim_vi_recon.imu_measurements = init_vi_recon.imu_measurements
    optim_vi_recon.write(optim_model_path)


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

    src = "estimate" if estimate is not None else "mps"

    # Estimate to Lamaria Reconstruction
    est_options = options.estimate_to_colmap_options
    if src == "estimate":
        assert estimate.exists(), (
            "Estimate path must be provided if not using MPS"
        )

        _ = run_estimate_to_lamaria(
            est_options,
            vrs,
            options.images_path,
            estimate,
            options.colmap_model_path,
        )
    else:
        assert mps_folder.exists(), (
            "MPS folder path must be provided if using MPS"
        )

        _ = run_mps_to_lamaria(
            est_options,
            vrs,
            options.images_path,
            mps_folder,
            options.colmap_model_path,
        )

    # Keyframe Selection
    kf_options = options.keyframing_options
    _ = run_keyframe_selection(
        kf_options,
        options.colmap_model_path,
        options.images_path,
        options.keyframes_path,
        options.kf_model_path,
    )

    # Triangulation
    tri_options = options.triangulator_options
    _ = run_triangulation(
        tri_options,
        options.kf_model_path,
        options.keyframes_path,
        options.hloc_path,
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
