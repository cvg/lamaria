<h1 align="center">Benchmarking Egocentric Visual-Inertial SLAM at City Scale</h2>

We present **LaMAria**, a egocentric, city-scale benchmark for **visual-inertial SLAM**, featuring 
~ **22 hours / 70 km** of trajectories with survey-grade control points providing **centimeter-accurate ground truth**.

Using **LaMAria**, you can:
- Evaluate SLAM systems under real-world challenges: low light, moving platforms, exposure changes, extremely long trajectories.
- Benchmark against highly accurate sparse ground truths.

<p align="center">
  <img src="assets/teaser_final.png" alt="Overview of LaMAria" width="900"/><br/>
  <em>Figure 1: Overview of the LaMAria dataset and benchmark.</em>
</p>

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
This repository supports Python 3.9 through 3.13. Installing the package `lamaria` pulls the other dependencies.

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


## Evaluation


### Evaluation w.r.t. Control Points


### Evaluation w.r.t Pseudo-GT


### EVO Evaluation w.r.t MPS


## Data Conversion



## Example Visual-Inertial Optimization



## BibTeX citation