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

### Downloading LaMAria

```bash
python tools/download_lamaria.py --help
```

To download the dataset conveniently, we provide a custom script `tools/download_lamaria.py`. Using this script, you can download:
- Specific sequences or entire sets (training/test).
- Specific types:
  - Raw - Downloads raw `.vrs` files and Aria calibration file.
  - Pinhole - Downloads ASL folder, rosbag, and pinhole calibration file.
  - Both - Downloads both raw and pinhole data.

For example, to download all training sequences with both raw and pinhole data, run:

```bash
python tools/download_lamaria.py --set training --type both
```
**Please note that the full archive is very large (~3.5 TB). Download full sets only if you have sufficient storage, else** 💣.

The training and test sequence information can be found in the <a href="https://lamaria.ethz.ch/slam_datasets" target="_blank" rel="noopener noreferrer">dataset page</a>.

To download the raw data of a specific sequence (e.g., `R_01_easy`), run:

```bash
python tools/download_lamaria.py --sequences R_01_easy --type raw
```
Ground truth files are available only for the training sequences.

To learn more about the various data formats, calibration files and ground-truths, please visit our <a href="https://lamaria.ethz.ch/slam_documentation" target="_blank" rel="noopener noreferrer">dataset documentation</a>.

## Evaluation


### Evaluation w.r.t. Control Points


### Evaluation w.r.t Pseudo-GT


### EVO Evaluation w.r.t MPS


## Data Conversion



## Example Visual-Inertial Optimization

### Additional Installation
To extract images from a `.vrs` file, it is required to install the VRS Command Line Tools. Please follow the instructions [here](https://github.com/facebookresearch/vrs?tab=readme-ov-file#instructions-macos-and-ubuntu-and-container) to install the library from source.



## BibTeX citation
