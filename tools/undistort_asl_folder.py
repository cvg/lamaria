import argparse
import copy
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pycolmap
from tqdm import tqdm

from lamaria.utils.aria import (
    initialize_reconstruction_from_calibration_file,
)
from lamaria.utils.constants import (
    ARIA_CAMERAS,
)


def undistort_reconstruction(
    image_path: Path,
    rec_path: Path,
    output_path: Path,
    ratio_blank_pixels: float = 0.2,
    verbose: bool = False,
) -> dict[int, pycolmap.Camera]:
    with tempfile.TemporaryDirectory() as tmp_path:
        tmp_path = Path(tmp_path)

        # Run COLMAP undistortion
        print(
            f"Undistorting the images to {tmp_path}, this may take a while..."
        )
        cmd = f"colmap image_undistorter --image_path {image_path} \
                --input_path {rec_path} --output_path {tmp_path} \
                --blank_pixels {ratio_blank_pixels}"
        stdout = None if verbose else subprocess.PIPE
        out = subprocess.run(
            cmd, shell=True, stderr=subprocess.STDOUT, stdout=stdout, check=True
        )
        if out.returncode:
            msg = f"Command '{cmd}' returned {out.returncode}."
            if out.stdout:
                msg += "\n" + out.stdout.decode("utf-8")
            raise subprocess.SubprocessError(msg)
        print("Done undistorting!")

        # Copy the cameras
        rec = pycolmap.Reconstruction(tmp_path / "sparse")
        cameras = {i: copy.copy(c) for i, c in rec.cameras.items()}

        if output_path.exists():
            if not output_path.is_dir() or output_path.iterdir():
                raise ValueError(
                    f"Output path {output_path} is not an empty directory."
                )
            output_path.rmdir()
        shutil.move(tmp_path / "images", output_path)

    return cameras


def write_cameras_json(cameras: dict[int | str, pycolmap.Camera], path: Path):
    camera_dicts = {}
    for key, c in cameras.items():
        d = d = c.todict()
        d["model"] = c.model.name
        d["params"] = c.params.tolist()
        camera_dicts[key] = d
    path.write_text(json.dumps(camera_dicts))


def undistort_asl(
    calibration_file: Path, asl_path: Path, output_asl_path: Path, **kwargs
):
    if output_asl_path.exists():
        raise ValueError(f"{output_asl_path=} already exists.")
    if not asl_path.exists():
        raise FileNotFoundError(f"{asl_path=}")
    if not calibration_file.exists():
        raise FileNotFoundError(f"{calibration_file=}")

    recon = initialize_reconstruction_from_calibration_file(
        calibration_file=calibration_file,
    )
    # Create a dummy COLMAP reconstruction.
    image_ext = ".png"
    image_id = 1
    rig = recon.rig(rig_id=1)
    colmap_images = {}
    for i, (key, _) in enumerate(ARIA_CAMERAS):
        cam = recon.cameras[i + 1]
        image_names = sorted(
            p.relative_to(asl_path)
            for p in (asl_path / "aria" / key).glob(f"**/*{image_ext}")
        )
        colmap_images[key] = []
        for n in image_names:
            im = pycolmap.Image(
                n,
                pycolmap.Point2DList(),
                cam.camera_id,
                image_id,
            )
            image_id += 1
            colmap_images[key].append(im)

    zipped_images = list(zip(*[colmap_images[key] for key, _ in ARIA_CAMERAS]))

    for j, (left_im, right_im) in enumerate(
        tqdm(
            zipped_images,
            total=len(zipped_images),
            desc="Adding images to reconstruction",
        )
    ):
        frame = pycolmap.Frame()
        frame.rig_id = rig.rig_id
        frame.frame_id = j + 1
        frame.rig_from_world = pycolmap.Rigid3d()
        left_im.frame_id = frame.frame_id
        right_im.frame_id = frame.frame_id
        frame.add_data_id(left_im.data_id)
        frame.add_data_id(right_im.data_id)
        recon.add_frame(frame)
        recon.add_image(left_im)
        recon.add_image(right_im)

    with tempfile.TemporaryDirectory() as temp_rec_path:
        recon.write(temp_rec_path)
        cameras_undist = undistort_reconstruction(
            asl_path, temp_rec_path, output_asl_path, **kwargs
        )

    # Copy the undistorted cameras in json.
    cameras_undist = {
        key: cameras_undist[i + 1] for i, (key, _) in enumerate(ARIA_CAMERAS)
    }
    write_cameras_json(
        cameras_undist, output_asl_path / "aria" / "cameras.json"
    )

    # Copy the other files
    for p in asl_path.glob("**/*"):
        if p.is_dir() or p.suffix == image_ext:
            continue
        p_out = output_asl_path / p.relative_to(asl_path)
        p_out.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(p, p_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration_file",
        type=Path,
        required=True,
        help="Path to Aria calibration json file",
    )
    parser.add_argument(
        "--asl_path",
        type=Path,
        required=True,
        help="Path to input Aria distorted ASL folder",
    )
    parser.add_argument(
        "--output_asl_path",
        type=Path,
        required=True,
        help="Path to output undistorted ASL folder",
    )
    parser.add_argument("--ratio_blank_pixels", type=float, default=0.2)
    args = parser.parse_args()

    undistort_asl(
        args.calibration_file,
        args.asl_path,
        args.output_asl_path,
        ratio_blank_pixels=args.ratio_blank_pixels,
    )
