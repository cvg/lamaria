import pycolmap
from copy import deepcopy
from typing import Optional
from pathlib import Path

from lamaria.rig.lamaria_reconstruction import LamariaReconstruction
from lamaria.rig.keyframing.to_colmap import EstimateToColmap
from lamaria.rig.keyframing.keyframe_selection import KeyframeSelector
from lamaria.rig.optim.triangulation import run as triangulation_run
from lamaria.rig.optim.session import SingleSeqSession
from lamaria.rig.optim.vi_optimization import run
from lamaria.rig.config.pipeline import PipelineOptions
from lamaria.rig.config.options import (
    EstimateToColmapOptions,
    KeyframeSelectorOptions,
    TriangulatorOptions
)


def run_estimate_to_colmap(
    options: EstimateToColmapOptions,
    vrs: Path,
    estimate: Path,
    images: Path,
    colmap_model: Path,
    mps_folder: Optional[Path] = None,
) -> LamariaReconstruction:

    options = options.set_custom_paths(
        vrs,
        estimate,
        images,
        colmap_model,
        mps_folder,
    )

    if colmap_model.exists():
        lamaria_recon = LamariaReconstruction.read(colmap_model)
        return lamaria_recon
    else:
        colmap_model.mkdir(parents=True, exist_ok=True)

    est_to_colmap = EstimateToColmap(options)
    lamaria_recon = est_to_colmap.create()
    lamaria_recon.write(colmap_model)

    return lamaria_recon


def run_keyframe_selection(
    options: KeyframeSelectorOptions,
    input: Path | LamariaReconstruction,
    images: Path,
    keyframes: Path,
    kf_model: Path,
) -> LamariaReconstruction:
    
    if isinstance(input, Path):
        input_recon = LamariaReconstruction.read(input)
    else:
        input_recon = input

    options = options.set_custom_paths(
        keyframes,
        kf_model,
    )

    if kf_model.exists():
        kf_lamaria_recon = LamariaReconstruction.read(kf_model)
        return kf_lamaria_recon
    else:
        kf_model.mkdir(parents=True, exist_ok=True)

    kf_selector = KeyframeSelector(options, input_recon)
    kf_lamaria_recon = kf_selector.run_keyframing()
    
    kf_lamaria_recon.write(kf_model)

    if not keyframes.exists():
        keyframes.mkdir(parents=True, exist_ok=True)

    kf_selector.copy_images_to_keyframes_dir(images, keyframes)

    return kf_lamaria_recon


def run_triangulation(
    options: TriangulatorOptions,
    input: Path, # path to LamariaReconstruction
    keyframes: Path,
    hloc: Path,
    pairs_file: Path,
    tri_model: Path,
) -> LamariaReconstruction:
    if not isinstance(input, Path):
        raise ValueError("Input must be a Path to the reconstruction")
    
    assert input.exists(), f"input reconstruction path {input} does not exist"

    options = options.set_custom_paths(
        hloc,
        pairs_file,
        tri_model,
    )

    triangulated_model_path = triangulation_run(
        options,
        reference_model=input,
        keyframes=keyframes,
    )

    input_lamaria_recon = LamariaReconstruction.read(input)
    tri_recon = pycolmap.Reconstruction(triangulated_model_path)

    tri_lamaria_recon = LamariaReconstruction()
    tri_lamaria_recon.reconstruction = tri_recon
    tri_lamaria_recon.timestamps = input_lamaria_recon.timestamps
    tri_lamaria_recon.imu_measurements = input_lamaria_recon.imu_measurements
    tri_lamaria_recon.write(triangulated_model_path)

    return tri_lamaria_recon


def run_pipeline():
    pipeline_options = PipelineOptions()

    # Step 1: Estimate to COLMAP
    est_options = pipeline_options.get_estimate_to_colmap_options()
    _ = run_estimate_to_colmap(
        est_options,
        est_options.vrs,
        est_options.estimate,
        est_options.images,
        est_options.colmap_model,
    )

    # Step 2: Keyframe Selection
    kf_options = pipeline_options.get_keyframing_options()
    _ = run_keyframe_selection(
        kf_options,
        est_options.colmap_model,
        est_options.images,
        kf_options.keyframes,
        kf_options.kf_model,
    )

    # Step 3: Triangulation
    tri_options = pipeline_options.get_triangulator_options()
    _ = run_triangulation(
        tri_options,
        kf_options.kf_model,
        kf_options.keyframes,
        tri_options.hloc,
        tri_options.pairs_file,
        tri_options.tri_model,
    )

if __name__ == "__main__":
    run_pipeline()
