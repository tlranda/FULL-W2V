# Writing Specification Files:

## Naming Convention:

All file names follow the format "\#\_<qualifier>", where \# is an integer, \_ is the underscore character,
and <qualifier> is a human-readable identifier that indicates when the specification is appropriate to use.

## Specifications:

Comments can be written in specification files on lines that START with the \# character.

The first non-comment line should be the integer priority of the specification. Specifications are loaded
in ascending \# order, so specifications with greater numbers can override parameters in lower \# files,
thus the file naming convention to make this information clearly available without opening files.

In a specification file, arguments can be given in the format "[\*]<argument>: <CSV>", where \* is an optional
character indicating that the argument's value is a file that should be checked for existence prior to running
performance evaluations. All files are expected to be given as absolute paths or relative to the location of
`sweep.py`. Each argument should be given with relevant '-' or '--' characters, using the ':' character as a
delimiter. All valid values should follow as comma-separated values on the same line. The comma separation is
based on naive comma delimiters; escaping comma characters within an argument value is not supported at this point.

Note that the `sweep.py` script EXHAUSTIVELY searches ALL combinations of ALL arguments across EACH specification
level, however multiple files defining the same specification level are exclusive with respect to one another.

