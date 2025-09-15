from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pycolmap

from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    triangulation
)

from ... import logger
from ..config.loaders import load_cfg


def pairs_from_frames(recon: pycolmap.Reconstruction):
    rig_pairs = set()
    by_index = defaultdict(list)

    for fid in sorted(recon.frames.keys()):
        fr = recon.frames[fid]
        img_ids = sorted([d.id for d in fr.data_ids])
        names = [recon.images[i].name for i in img_ids]

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                rig_pairs.add((names[i], names[j]))
                rig_pairs.add((names[j], names[i]))

        for k, n in enumerate(names):
            by_index[k].append(n)

    adj_pairs = set()
    for idx, seq in by_index.items():
        for a, b in zip(seq[:-1], seq[1:]):
            adj_pairs.add((a, b))

    return rig_pairs, adj_pairs

def postprocess_pairs_with_reconstruction(
    sfm_pairs_file: Path,
    reconstruction: pycolmap.Reconstruction | Path
):
    recon = (reconstruction if isinstance(reconstruction, pycolmap.Reconstruction)
             else pycolmap.Reconstruction(str(reconstruction)))

    rig_pairs, adj_pairs = pairs_from_frames(recon)

    existing = set()
    with open(sfm_pairs_file, "r") as f:
        for line in f:
            a, b = line.strip().split()
            existing.add((a, b))

    existing = {p for p in existing if p not in rig_pairs}
    existing |= adj_pairs

    with open(sfm_pairs_file, "w") as f:
        for a, b in sorted(existing):
            f.write(f"{a} {b}\n")


def run(
    cfg=None,
    num_retrieval_matches: int = 5,
) -> Path:
    
    cfg = load_cfg() if cfg is None else cfg

    keyframes_dir = cfg.result.output_folder_path / cfg.result.keyframes
    if not keyframes_dir.exists():
        raise FileNotFoundError(f"keyframes_dir not found at {keyframes_dir}")
    
    hloc_outputs_dir = cfg.result.output_folder_path / "hloc"
    hloc_outputs_dir.mkdir(parents=True, exist_ok=True)

    reference_model_path = cfg.result.output_folder_path / cfg.result.kf_model
    if not reference_model_path.exists():
        raise FileNotFoundError(f"reference_model not found at {reference_model_path}")

    triangulated_model_path = cfg.result.output_folder_path / cfg.result.tri_model
    pairs_path = hloc_outputs_dir / cfg.triangulation.pairs_file

    retrieval_conf = extract_features.confs[cfg.triangulation.retrieval_conf]
    feature_conf   = extract_features.confs[cfg.triangulation.feature_conf]
    matcher_conf   = match_features.confs[cfg.triangulation.matcher_conf]

    logger.info("HLOC confs: retrieval=%s, features=%s, matcher=%s",
                cfg.triangulation.retrieval_conf,
                cfg.triangulation.feature_conf,
                cfg.triangulation.matcher_conf)

    retrieval_path = extract_features.main(retrieval_conf, image_dir=keyframes_dir, export_dir=hloc_outputs_dir)
    features_path = extract_features.main(feature_conf, image_dir=keyframes_dir, export_dir=hloc_outputs_dir)

    pairs_from_retrieval.main(retrieval_path, pairs_path, num_retrieval_matches)
    postprocess_pairs_with_reconstruction(pairs_path, reference_model_path)

    matches_path = match_features.main(
        conf=matcher_conf,
        pairs=pairs_path,
        features=feature_conf["output"],
        export_dir=hloc_outputs_dir,
    )

    triangulated_model = triangulation.main(
        sfm_dir=triangulated_model_path,
        reference_model=reference_model_path,
        image_dir=keyframes_dir,
        pairs=pairs_path,
        features=features_path,
        matches=matches_path,
    )

    return triangulated_model