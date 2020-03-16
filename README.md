# Chance-Constrained Sequential Convex Programming

This repository accompanies our ECC 2020 paper *[Chance-Constrained Sequential Convex Programming for Robust Trajectory Optimization](http://asl.stanford.edu/wp-content/papercite-data/pdf/Lew.Bonalli.Pavone.ECC20.pdf)*: an algorithm for uncertainty-aware trajectory planning.


<p align="center">
  <img src="doc/freeflyer.png" width="50%"/>
  <br /><em>Free-flyer robot avoiding obstacles despite uncertain dynamics.</em>
</p>


## Setup

Python 3.5.2 is required. It is advised to run the following commands within a virtual environment. 
```bash
	python -m venv ./venv
	source venv/bin/activate
```
Then, install the package as
```bash
	pip install -r requirements.txt
	pip install -e .
```

## Demo

We provide examples for an uncertain free-flyer system and for the *[Astrobee](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20160007769.pdf)* robot navigating on-board the International Space Station.
```bash
	cd exps/
	python freeflyer_script.py
	python astrobee_script.py
```

### Demo
We report computation times below, measured on a laptop equipped with an Intel Core i7-6700 CPU at 2.60GHz with 8GB of RAM.

| Computation Time (Nb. SCP Iters) / Average |  N = 35 | N = 30 | N = 25 | N = 20 |
| :---: | :---: | :---: | :---: | :---: |
| Freeflyer (nonlinear., CoM offset, 4 obstacles) | 0.174s (3) / 58ms | 0.152s (3) / 51ms | 0.126s (3) / 42ms | 0.104s (3) / 35ms |
| Astrobee (ISS environment with 30 obstacles) | 3.66s (6) / 0.61s | 2.40s (5) / 0.48s | 1.91s (5) / 0.38s | 1.53s (5) / 0.31s |


## Citation

```
@inproceedings{lew2020ccscp,
  title={Chance-Constrained Sequential Convex Programming for Robust Trajectory Optimization},
  author={Lew, Thomas and Bonalli, Riccardo and Pavone, Marco},
  booktitle={European Control Conference},
  year={2020}
}
```
