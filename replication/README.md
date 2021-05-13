# Organization and Usage

`template.pbs` provides a basic script (also SLURM compatible) for automating evaluation of Word2Vec implementations.
Several such evaluations can be queued up at once using `run.pbs`, but note that evaluations are serial (not distributed).
You will likely need to edit module loads and PBS/SLURM node selection/allocation parameters.

`sweep.py` is the evaluation driver, which is invoked by `template.pbs`.
The Word2Vec parameters are written and stored in `evaluation`, such that `#_qualifier.txt` represents a hierarchical specification.
The run and template scripts use corpus, system, and implementation information as qualifiers with respective ranks 1, 2, and 3
to dynamically add all requirements in these specification files to the runtime arguments used in `sweep.py` for fair and accurate evaluations.

`FULL-text8.sh` provides a simple script for use in the FULL-W2V quickstart.
