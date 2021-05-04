# FULL-W2V: Fully Exploiting Data Reuse for W2V on GPU-Accelerated Systems
## SCALAB Team: Thomas Randall, Tyler Allen, Dr. Rong Ge

FULL-W2V represents state-of-the-art throughput on Word2Vec using heterogeneous GPU systems.
It supports the "Independence of Negative Samples" and "Lifetime Reuse of Context Words" described in the ICS paper ["FULL-W2V: Fully Exploiting Data Reuse for W2V on GPU-Accelerated Systems"](https://doi.org/10.1145/3447818.3460373).

# LICENSE
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

# Setup and Installation

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

## In This Repository:

`makefile`: Compiles code for all implementations or specific ones as in `make <name>`. GDB debugging available if supported via `make gdb_<name>`. Common references for all makefiles in the repository are sourced from `common.inc`.

`src`: All evaluated implementations. Scripts are provided to retrieve and install related works to reduce the download size of this repository.

`data`: Includes scripts to download datasets used in the paper, but not directly included in the repository.

`analysis`: Counter-based and postmortem analysis scripts.

`replication`: Scripts to aid replicating paper experiments.

## Copyright:

ICS '21, June 14–17, 2021, Virtual Event, USA

© 2021 Copyright is held by the owner/author(s). Publication rights licensed to ACM.

ACM ISBN 978-1-4503-8335-6/21/06...$15.00 [https://doi.org/10.1145/3447818.3460373](https://doi.org/10.1145/3447818.3460373)

## Funding:

Funded by NSF (National Science Foundation) award numbers: CCF-1551511, CNS-1551262

