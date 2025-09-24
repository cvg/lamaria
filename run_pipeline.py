import pycolmap
import shutil
from typing import Optional
from pathlib import Path

from lamaria.rig.lamaria_reconstruction import LamariaReconstruction
from lamaria.rig.keyframing.to_colmap import EstimateToColmap
from lamaria.rig.keyframing.keyframe_selection import KeyframeSelector
from lamaria.rig.optim.triangulation import run as triangulate
from lamaria.rig.optim.session import SingleSeqSession
from lamaria.rig.optim.vi_optimization import run as run_vi_optimization
from lamaria.rig.config.pipeline import PipelineOptions
from lamaria.rig.config.options import (
    EstimateToColmapOptions,
    KeyframeSelectorOptions,
    TriangulatorOptions,
    VIOptimizerOptions,
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
    
    if tri_model.exists():
        tri_lamaria_recon = LamariaReconstruction.read(tri_model)
        return tri_lamaria_recon

    triangulated_model_path = triangulate(
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


def run_optimization(
    options: VIOptimizerOptions,
    input: Path, # path to LamariaReconstruction
    optim_model: Path,
) -> LamariaReconstruction:
    if not isinstance(input, Path):
        raise ValueError("Input must be a Path to the reconstruction")
    
    db_path = input / "database.db"
    assert db_path.exists(), f"Database path {db_path} does not exist in input reconstruction"
    if optim_model.exists():
        optim_lamaria_recon = LamariaReconstruction.read(optim_model)
        return optim_lamaria_recon
    
    optim_model.mkdir(parents=True, exist_ok=True)
    db_dst = optim_model / "database.db"
    shutil.copy(db_path, db_dst)
    
    init_lamaria_recon = LamariaReconstruction.read(input)
    session = SingleSeqSession(
        options,
        init_lamaria_recon,
    )

    optimized_recon = run_vi_optimization(
        session,
        db_dst,
    )

    optim_lamaria_recon = LamariaReconstruction()
    optim_lamaria_recon.reconstruction = optimized_recon
    optim_lamaria_recon.timestamps = init_lamaria_recon.timestamps
    optim_lamaria_recon.imu_measurements = init_lamaria_recon.imu_measurements
    optim_lamaria_recon.write(optim_model)


def run_pipeline():
    pipeline_options = PipelineOptions()

    # Estimate to COLMAP
    est_options = pipeline_options.get_estimate_to_colmap_options()
    _ = run_estimate_to_colmap(
        est_options,
        est_options.vrs,
        est_options.estimate,
        est_options.images,
        est_options.colmap_model,
    )

    # Keyframe Selection
    kf_options = pipeline_options.get_keyframing_options()
    _ = run_keyframe_selection(
        kf_options,
        est_options.colmap_model,
        est_options.images,
        kf_options.keyframes,
        kf_options.kf_model,
    )

    # Triangulation
    tri_options = pipeline_options.get_triangulator_options()
    _ = run_triangulation(
        tri_options,
        kf_options.kf_model,
        kf_options.keyframes,
        tri_options.hloc,
        tri_options.pairs_file,
        tri_options.tri_model,
    )

    # Visual-Inertial Optimization
    vi_options = pipeline_options.get_vi_optimizer_options()
    _ = run_optimization(
        vi_options,
        tri_options.tri_model,
        vi_options.optim.optim_model,
    )

if __name__ == "__main__":
    run_pipeline()
