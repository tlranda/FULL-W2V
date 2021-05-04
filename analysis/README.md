# HyperWords Implementation
hyperwords: [OmerLevy Hyperwords repository](https://bitbucket.org/omerlevy/hyperwords), since deactivated

evalHyperWords.py: Provides a convenient script to operate the OmerLevy Hyperwords evaluations

remerge.py: Recovers original embedding files from .vocab & .npy files

# Counter Analyses:

## NSIGHT
### Setup
`nsight.sh`: Write the path to your `nv-nsight-cu-cli` on line 12 "NSIGHT\_EXE=<path>".

### Usage:
`nsight.sh`: ./nsight.sh \<output\> \<implementation\>
* \<output\> is the current working directory prefix for the desired .words embedding output.
* \<implementation\> is the name of the implementation to profile.

## NVPROF
### Usage:
`nvprof.sh`: ./nvprof.sh \<output\> \<implementation\>
* \<output\> is the current working directory prefix for the desired .words embedding output.
* \<implementation\> is the name of the implementation to profile.

