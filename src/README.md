# Code Organization and Installation Management:
The FULL-W2V implementation is included under the directory with the same name.

Other works can be installed using the provided installer scripts located in `Install`.

Use `Install/install_all.sh` to run all installer scripts, or `Install/install_<implementation>.sh` to install a specific implementation.

Use `Install/uninstall.sh` to remove all directories created by the previous install script.

Each installer script performs the following:
* Clone the respective repository
* Check out the specific commit used at the time of publication
* Remove duplicated or redundant files included in the commit
* Perform several patches for compatibility with various scripts used in this repository
* Invoke the implementation's installation process, if it provides one

# Patches:
The included patches in the `Install/Patch` directory DO NOT modify core operability of works, and can
be undone using `Install/Patch/<implementation>/./unpatch.sh`. Any files deleted during the
installation process may be retrieved using Git.

# Other Data:
For trainable data to use with all implementations, see the `Data` directory at the top level of the repository.

