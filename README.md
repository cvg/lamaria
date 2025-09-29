<h1 align="center">Benchmarking Egocentric Visual-Inertial SLAM at City Scale</h2>

We present **LaMAria**, an egocentric, city-scale benchmark for **visual-inertial SLAM**, featuring 
~ **22 hours / 70 km** of trajectories with survey-grade control points providing **centimeter-accurate ground truth**.

Using **LaMAria**, you can:
- Evaluate SLAM systems under real-world challenges: low light, moving platforms, exposure changes, extremely long trajectories, and challenges typical to egocentric motion.
- Benchmark against highly accurate ground truths.

This dataset offers 23 training sequences and 63 test sequences. 

<p align="center">
  <img src="assets/teaser_final.png" alt="Overview of LaMAria" width="900"/><br/>
  <em>Figure 1: Overview of the LaMAria dataset and benchmark.</em>
</p>

In this repository, you can find scripts to conveniently download the dataset, evaluate SLAM results, perform data conversions, and run an example visual-inertial optimization pipeline.

To learn more about the dataset, please visit our <a href="https://lamaria.ethz.ch" target="_blank" rel="noopener noreferrer">main dataset website</a> or read our <a href="" target="_blank" rel="noopener noreferrer">paper</a>.

## Table of Contents
- [Installation](#installation)
- [Downloading the Dataset](#downloading-the-dataset)
- [Evaluation](#evaluation)
  - [Evaluation w.r.t. Control Points](#evaluation-wrt-control-points)
  - [Evaluation w.r.t. Pseudo-GT](#evaluation-wrt-pseudo-gt)
  - [EVO Evaluation w.r.t. MPS](#evo-evaluation-wrt-mps)
- [Data Conversion](#data-conversion)
- [Example Visual-Inertial Optimization](#example-visual-inertial-optimization)
- [BibTeX Citation](#bibtex-citation)


## Installation
This repository supports Python 3.9 through 3.14. Installing the package `lamaria` pulls other dependencies
mentioned in `requirements.txt`.

Create an environment:
```bash
python3 -m venv lamaria_env
source lamaria_env/bin/activate
```

Clone the repository:
```bash
git clone git@github.com:cvg/lamaria.git
cd lamaria
pip install -r requirements.txt
```

Install the package:
```bash
python -m pip install -e .
```

## Downloading the dataset
Our dataset is fully hosted via the archive <a href="https://cvg-data.inf.ethz.ch/lamaria/" target="_blank" rel="noopener noreferrer">here</a>.

### Quickstart
We provide a small script `quickstart.sh` that downloads one sequence from the archive. The standalone evaluations and example visual-inertial optimization can be run on the downloaded demo data.

```bash
chmod +x quickstart.sh
./quickstart.sh
```

The data is stored in the `demo/` folder. You may run the standalone evaluations and example visual-inertial optimization on this data.

### Downloading LaMAria

```bash
python tools/download_lamaria.py --help
```

For download convenience, we provide a custom script `tools/download_lamaria.py`. Using this script, you can download:
- Specific sequences or entire sets (training/test).
- Specific types:
  - Raw - Downloads raw `.vrs` files and Aria calibration file.
  - Pinhole - Downloads ASL folder, rosbag, and pinhole calibration file.
  - Both - Downloads both raw and pinhole data.

Ground truth files are automatically downloaded for the training sequences. 

**💣 Please note that the full archive is very large (~3.5 TB). Download full sets only if you have sufficient storage 💣.**

#### Some example commands

To download all training sequences in both raw and pinhole formats:
```bash
python tools/download_lamaria.py --set training --type both
```
To download the raw data of a specific sequence (e.g., `R_01_easy`):
```bash
python tools/download_lamaria.py --sequences R_01_easy --type raw
```
To download 3 custom sequences in pinhole format:
```bash
python tools/download_lamaria.py --sequences sequence_1_1 sequence 1_2 sequence 1_3 --type pinhole
```

#### Output structure
The downloaded data is stored in the following way:
```
out_dir/
└── lamaria/
    ├── training/
    │   ├── R_01_easy/
    │   │   ├── aria_calibrations/
    │   │   │   └── R_01_easy.json
    │   │   ├── asl_folder/
    │   │   │   └── R_01_easy.zip
    │   │   ├── ground_truth/
    │   │   │   ├── pseudo_dense/
    │   │   │   │   └── R_01_easy.txt
    │   │   │   └── sparse/
    │   │   │       └── # if sequence has CPs
    │   │   ├── pinhole_calibrations/
    │   │   │   └── R_01_easy.json
    │   │   ├── raw_data/
    │   │   │   └── R_01_easy.vrs
    │   │   └── rosbag/
    │   │       └── R_01_easy.bag
    │   └── ...
    └── test/
        └── sequence_1_1
```

For more information about the training and test sequences, refer to the <a href="https://lamaria.ethz.ch/slam_datasets" target="_blank" rel="noopener noreferrer">dataset details</a>. To learn more about the various data formats, calibration files and ground-truths, visit our <a href="https://lamaria.ethz.ch/slam_documentation" target="_blank" rel="noopener noreferrer">documentation</a>.

## Evaluation
Our training and test sequences are categorized into varying challenges. To evaluate your SLAM results on our data, we provide two main ways:
1. **Evaluation via the website**: Upload your results on our <a href="https://lamaria.ethz.ch/login" target="_blank" rel="noopener noreferrer">website</a> to get evaluation results. Results on test sequences are displayed on the public leaderboard.
2. **Standalone evaluation**: Run the evaluation scripts locally using the provided `lamaria` package.

### Evaluation w.r.t. Control Points
Those sequences that observe ground truth control points (CPs) can be evaluated w.r.t. these points. This script computes the score and control point recall based on the alignment of the estimated trajectory to the control points.

To perform the evaluation on the downloaded demo data:
```bash
python -m evaluate_wrt_control_points --estimate demo/estimate/R_11_5cp.txt --cp_json_file demo/cp_data/R_11_5cp.json --device_calibration_json demo/calibrations/R_11_5cp.json --output_path demo/eval_cp --corresponding_sensor imu
```

*This command evaluates the provided estimate w.r.t. control points and stores the results in `demo/eval_cp`. The `--corresponding_sensor` flag indicates which sensor the poses are expressed in (e.g., `imu` or `cam0`).*

To learn more about the control points and sparse evaluation, refer to Section 4.1 and 4.2 of our <a href="" target="_blank" rel="noopener noreferrer">paper</a>.

### Evaluation w.r.t Pseudo-GT


### EVO Evaluation w.r.t MPS


## Data Conversion



## Example Visual-Inertial Optimization

### Additional Installation
To extract images from a `.vrs` file, it is required to install the VRS Command Line Tools. Please follow the instructions [here](https://github.com/facebookresearch/vrs?tab=readme-ov-file#instructions-macos-and-ubuntu-and-container) to install the library from source.



## BibTeX citation
