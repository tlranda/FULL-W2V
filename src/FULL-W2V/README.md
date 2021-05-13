# FULL-W2V: Fully Exploiting Data Reuse for W2V on GPU-Accelerated Systems

FULL-W2V is a highly optimized implementation of Word2Vec for designed for use with NVIDIA GPUs.
It provides It supports the "Independence of Negative Samples" and "Lifetime Reuse of Context Words" described in the ICS paper ["FULL-W2V: Fully Exploiting Data Reuse for W2V on GPU-Accelerated Systems"](https://doi.org/10.1145/3447818.3460373).

## LICENSE
Copyright 2021 Thomas Lenzy Randall

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the LICENSE for the specific language governing permissions and
limitations under the License.

## Setup and Installation

* Software Dependencies:
	+ The included make files support CUDA with g++
	+ The included Python3 scripts depend on the following modules:
		* [Matplotlib](https://matplotlib.org/)
		* [Numpy](https://numpy.org/)
		* [Pandas](https://pandas.pydata.org/)
	+ The included Python2 scripts depend on the following modules:
		* [Numpy](https://numpy.org/)
		* [docopt](https://github.com/docopt/docopt)
		* [scipy](https://www.scipy.org/)
	+ For CPU-based related works, MKL libraries and the Intel `icpc` compiler are necessary

## Quick Start

1. Download the code: `git clone git@github.com:tlranda/FULL-W2V.git`.
2. Download corpus files using the provided script `data/./setup.sh`.
3. [Optional] Install related works using `src/Install/./install_all.sh` or `src/Install/./install_<implementation>.sh`.
4. Run `make FULL-W2V` to compile the binary. Other make targets are provided for each implementation included from 3.
5. For improved I/O and to utilize the scripts in the next optional step, run `<FULL-W2V> -train data/corpus/<corpus> -save-vocab data/vocab/<corpus>.vocab`, which enables use of `-read-vocab data/vocab/<corpus>.vocab` across all implementations.
6. [Optional] The `replication` directory contains scripts for replicating the paper's throughput and embedding quality evaluations. Further instructions included in the README of `replication`, but running `replication/FULL-text8.sh` should demonstrate FULL-W2V's performance.
7. Running any binary without any arguments will produce detailed option descriptions to run the implementation as you please.
