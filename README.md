<!---

	Copyright (c) 2009, 2018 Robert Bosch GmbH and its subsidiaries.
	This program and the accompanying materials are made available under
	the terms of the Bosch Internal Open Source License v4
	which accompanies this distribution, and is available at
	http://bios.intranet.bosch.com/bioslv4.txt

-->

[![Build Status](https://hi-z0bfk.hi.de.bosch.com:9080/buildStatus/icon?job=AEC4%2FBOLT-Pipeline-Master)](https://hi-z0bfk.hi.de.bosch.com:9080/job/AEC4/job/BOLT-Pipeline-Master/) [![License: BIOSL v4](http://bios.intranet.bosch.com/bioslv4-badge.svg)](#license)

<a name="bosch-learning-toolbox-bolt"></a>

# Bosch Learning Toolbox (BOLT)

<a name="contents"></a>

## Contents

- [Bosch Learning Toolbox (BOLT)](#bosch-learning-toolbox-bolt)
	- [Contents](#contents)
	- [The Bosch Learning Toolbox - A Short Introduction](#the-bosch-learning-toolbox---a-short-introduction)
	- [The BOLT Documentation](#the-bolt-documentation)
	- [Quick Installation](#quick-installation)
		- [Install BOLT](#install-bolt)
		- [BOLT with Visual Studio Code](#bolt-with-visual-studio-code)
		- [Update BOLT](#update-bolt)
		- [Install other packages](#install-other-packages)
	- [About](#about)
		- [3rd Party Licenses](#3rd-party-licenses)
		- [License](#license)

-------------------------------

<a name="the-bosch-learning-toolbox---a-short-introduction"></a>

## The Bosch Learning Toolbox - A Short Introduction

The Bosch Learning Toolbox (BOLT) is a python library for neural network training.
It extends [tf.keras](https://www.tensorflow.org/guide/keras),
TensorFlow's high-level API for building and training deep learning models, with additional functionality.
While tf.keras provides the basic building blocks for defining and optimizing neural networks,
BOLT provides more application-driven code.
As of now, it is especially tailored to computer vision tasks.

Bosch Learning Toolbox

- Configurable training protocols for training neural networks
- Code for reading various data formats for training and testing purposes
- Application-specific code, e.g. for object detection or semantic segmentation
- Extension to the Keras API, such as layer types, optimization algorithms or metrics that are originally not part of Keras
- Interfaces to communicate with other processes (e.g. for online demonstrators)

Most of the BOLT functionality is dependent on
[TensorFlow](https://github.com/tensorflow/tensorflow).
It additionally depends on some common Python packages.
BOLT is compatible with **Python 3.8 or later**. It does not run with **Python 2.x**!

In case you have questions or want to contribute to BOLT in general, feel free to join our
[**Team in Microsoft Teams**](https://teams.microsoft.com/l/team/19%3a72c001b27fb14e6b9e82d2ec4155e03e%40thread.tacv2/conversations?groupId=51faca56-85e7-45f8-8cf6-5ac818e4a480&tenantId=0ae51e19-07c8-4e4b-bb6d-648ee58410f4)
, either using the [link](https://teams.microsoft.com/l/team/19%3a72c001b27fb14e6b9e82d2ec4155e03e%40thread.tacv2/conversations?groupId=51faca56-85e7-45f8-8cf6-5ac818e4a480&tenantId=0ae51e19-07c8-4e4b-bb6d-648ee58410f4)
or the code `vhe63p5`.

-------------------------------

<a name="the-bolt-documentation"></a>

## The BOLT Documentation

The source for the documentation of the toolbox is in the folder `docs`.
Follow the instructions in the file [`docs/README.md`](docs/README.md) to create an html documentation from the source code.

The compiled documentation for the `master` branch of the toolbox can be found [here](https://hi-z0bfk.hi.de.bosch.com/BOLT/Documentation/).

A detailed installation guide can be found in the [community wiki](https://connect.bosch.com/wikis/home?lang=de-de#!/wiki/W3eb0d044ffb7_45d9_985b_d75c2d7a7dad/page/Installation%20Guide).

-------------------------------

<a name="quick-installation"></a>

## Quick Installation

<a name="install-bolt"></a>

### Install BOLT

The instructions below assume a Linux environment. A Windows-based environment may work, but is not supported.

First make sure that you have access to the internet. During the installation of BOLT, some packages need to be
downloaded from the internet, so the installation will fail if no internet access is available. You could try a command
like `wget github.com` to download a web page and check if the internet connection works.

Start a terminal, create a virtual environment (e.g., with name `bolt`) for the toolbox and activate it.

```bash
python -m venv /<some_path>/bolt
source /<some_path>/bolt/bin/activate
```

Go to the directory of BOLT

```bash
cd <bolt_directory>
```

From this directory, you can install BOLT packages using the ``bolt-pkg.sh`` script.

```bash
./bolt-pkg.sh --add default
```

This installs the default selection of BOLT packages in editable mode, which means that any changes to
your working copy are used directly when running BOLT, even without re-installing them. It also installs
the dependencies and development dependencies of BOLT in the versions that have been tested. The packages to
install are user configurable and can be one of:

- `default`: All BOLT packages, except `bolt-horovod`
- `all`: All BOLT packages, including `bolt-horovod`
- A custom selection of packages, e.g. `./bolt-pkg.sh --add bolt-core bolt-db`

If `horovod` is installed (e.g. by selecting `all` or `bolt-horovod`), `bolt-pkg.sh` will rebuild its binaries
for the installed version of TensorFlow, PyTorch, MPI and NCCL.

After installation, please set up pre-commit-hooks to ensure some code uniformity, for example removing whitespaces
at EOL. To activate the hooks, run this command after installation:

```bash
pre-commit install
```

Hooks can be manually run on the complete repository with the following command:

```bash
pre-commit run --all-files
```

Please note that pre-commit requires a Git version `>=2.17.x`. The often installed version `1.8.3.1` will not work.

<a name="bolt-with-visual-studio-code"></a>

### BOLT with Visual Studio Code


The recommended settings for Visual Studio Code can be found under the `.vscode` directory. To use them, copy any of the recommended files to their
counterparts used by VS Code, e.g.
- `launch_recommended.json` to `launch.json`
- `settings_recommended.json` to `settings.json`

The `extensions.json` file is used by VS Code to recommend useful extensions
the first time the BOLT folder is opened.

<a name="update-bolt"></a>

### Update BOLT

Activate the virtual environment you installed BOLT into (e.g., `bolt`)

```bash
source /<some_path>/bolt/bin/activate
```

Go to the directory with the updated version of BOLT

```bash
cd <bolt_directory>
```

Update the BOLT packages and dependencies

```bash
./bolt-pkg.sh --update
```

This will update any already installed BOLT packages and their dependencies to the versions tested with BOLT.

Please note that due to the use of pinned (exact) versions, some of them might conflict with third party packages
you might have installed manually in your virtual environment. In this case, `bolt-pkg` will display a warning
by `pip`, listing the conflicts. You can choose to resolve the conflicts by installing other versions of those
dependencies manually. As long as no conflicts to BOLT packages are shown (e.g.
`bolt-core requires xyz>=2 but you have xyz==1.1`), the environment should work. However, when deviating from the
pinned dependency versions, it is not guaranteed to work.


<a name="install-other-packages"></a>

### Install other packages

If you want to install other packages in your environment, it is recommended that you pin all common dependencies
to the versions that are used by and have been tested with BOLT.

```bash
# Install a 3rd party package, in this case `dash` as an example
pip install -c constraints.txt dash
```

In case of conflicts, this command will fail and `pip` will print a message with the incompatible packages.
Omitting the BOLT constraints and retrying the install command will only consider the dependency version
*ranges* defined for each package and not the exact pinned versions from `constraints.txt`:

```bash
# Install a 3rd party package, without constraints
pip install dash
```

Should this last command succeed without warnings, it is highly likely but not certain that the combination of BOLT
with the external packages will work. Please report any failures in the BOLT issue tracker to allow adjusting the
ranges of supported dependency versions.


<a name="about"></a>

## About

<a name="3rd-party-licenses"></a>

### 3rd Party Licenses


| Name                                                                                         | License            |
| -------------------------------------------------------------------------------------------- | ------------------ |
| [Gason](https://github.com/vivkin/gason)                                                     | MIT License        |
| [deep-learning-models](https://github.com/fchollet/deep-learning-models)                     | MIT License        |
| [freeze_session](https://github.com/Tony607/keras-tf-pb)                                     | MIT License        |
| [keras-applications](https://github.com/Callidior/keras-applications)                        | MIT License        |
| [keras-extras](https://github.com/kuza55/keras-extras)                                       | Apache License 2.0 |
| [keras](https://github.com/keras-team/keras)                                                 | Apache License 2.0 |
| [tensorflow-models](https://github.com/tensorflow/models)                                    | Apache License 2.0 |
| [ssd_keras](https://github.com/rykov8/ssd_keras)                                             | MIT License        |
| [tensorflow-triplet-loss](https://github.com/omoindrot/tensorflow-triplet-loss)              | MIT License        |
| [spatial-transformer-tensorflow](https://github.com/daviddao/spatial-transformer-tensorflow) | MIT License        |
| [RegNet](https://github.com/facebookresearch/pycls)                                          | MIT License        |
| [tensorflow-addons](https://github.com/tensorflow/addons)                                    | Apache License 2.0 |

For more details and the full text of the licenses, please see the documents under [legal](legal)

<a name="license"></a>

### License

> Copyright (c) 2009, 2018 Robert Bosch GmbH and its subsidiaries.
> This program and the accompanying materials are made available under
> the terms of the Bosch Internal Open Source License v4
> which accompanies this distribution, and is available at
> http://bios.intranet.bosch.com/bioslv4.txt

[![License: BIOSL v4](http://bios.intranet.bosch.com/bioslv4-badge.svg)](#license)
