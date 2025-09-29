from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pycolmap
from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    triangulation,
)

from ... import logger
from ...config.options import TriangulatorOptions


def get_colmap_triangulation_options(
    options: TriangulatorOptions,
) -> pycolmap.IncrementalPipelineOptions:
    colmap_options = pycolmap.IncrementalPipelineOptions()
    colmap_options.triangulation.merge_max_reproj_error = (
        options.merge_max_reproj_error
    )
    colmap_options.triangulation.complete_max_reproj_error = (
        options.complete_max_reproj_error
    )
    colmap_options.triangulation.min_angle = options.min_angle

    colmap_options.mapper.filter_max_reproj_error = (
        options.filter_max_reproj_error
    )
    colmap_options.mapper.filter_min_tri_angle = options.filter_min_tri_angle

    return colmap_options


def pairs_from_frames(recon: pycolmap.Reconstruction):
    frame_pairs = set()
    by_index = defaultdict(list)

    for fid in sorted(recon.frames.keys()):
        fr = recon.frames[fid]
        img_ids = sorted([d.id for d in fr.data_ids])
        names = [recon.images[i].name for i in img_ids]

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                frame_pairs.add((names[i], names[j]))
                frame_pairs.add((names[j], names[i]))

        for k, n in enumerate(names):
            by_index[k].append(n)

    adj_pairs = set()
    for _, seq in by_index.items():
        for a, b in zip(seq[:-1], seq[1:]):
            adj_pairs.add((a, b))

    return frame_pairs, adj_pairs


def postprocess_pairs_with_reconstruction(
    sfm_pairs_file: Path, reconstruction: pycolmap.Reconstruction | Path
):
    recon = (
        reconstruction
        if isinstance(reconstruction, pycolmap.Reconstruction)
        else pycolmap.Reconstruction(reconstruction.as_posix())
    )

    frame_pairs, adj_pairs = pairs_from_frames(recon)

    existing = set()
    with open(sfm_pairs_file) as f:
        for line in f:
            a, b = line.strip().split()
            existing.add((a, b))

    existing = {p for p in existing if p not in frame_pairs}
    existing |= adj_pairs

    with open(sfm_pairs_file, "w") as f:
        for a, b in sorted(existing):
            f.write(f"{a} {b}\n")


def run(
    options: TriangulatorOptions,
    reference_model: Path,  # reconstruction path
    keyframes_path: Path,
    output_path: Path,
) -> Path:
    if not keyframes_path.exists():
        raise FileNotFoundError(f"keyframes not found at {keyframes_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    hloc_output_path = output_path / "hloc"
    pairs_path = hloc_output_path / "pairs.txt"
    triangulated_model_path = output_path / "model"

    hloc_output_path.mkdir(parents=True, exist_ok=True)
    if not reference_model.exists():
        raise FileNotFoundError(
            f"reference_model not found at {reference_model}"
        )

    retrieval_conf = extract_features.confs[options.retrieval_conf]
    feature_conf = extract_features.confs[options.feature_conf]
    matcher_conf = match_features.confs[options.matcher_conf]

    logger.info(
        "HLOC confs: retrieval=%s, features=%s, matcher=%s",
        options.retrieval_conf,
        options.feature_conf,
        options.matcher_conf,
    )

    retrieval_path = extract_features.main(
        retrieval_conf, image_dir=keyframes_path, export_dir=hloc_output_path
    )
    features_path = extract_features.main(
        feature_conf, image_dir=keyframes_path, export_dir=hloc_output_path
    )

    pairs_from_retrieval.main(
        retrieval_path, pairs_path, options.num_retrieval_matches
    )
    postprocess_pairs_with_reconstruction(pairs_path, reference_model)

    matches_path = match_features.main(
        conf=matcher_conf,
        pairs=pairs_path,
        features=feature_conf["output"],
        export_dir=hloc_output_path,
    )

    colmap_opts = get_colmap_triangulation_options(options)

    _ = triangulation.main(
        sfm_dir=triangulated_model_path,
        reference_model=reference_model,
        image_dir=keyframes_path,
        pairs=pairs_path,
        features=features_path,
        matches=matches_path,
        mapper_options=colmap_opts,
    )

    return triangulated_model_path
