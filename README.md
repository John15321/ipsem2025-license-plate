[![semantic-release](https://img.shields.io/badge/semantic--release-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)

# IPSEM2025 License Plate

IPSEM 2025 License Plate detection.

# Features

The package comes with several modules for license plate recognition:

- **QNet**: Hybrid quantum-classical neural network for character recognition ([Documentation](src/ipsem2025_license_plate/qnet/README.md))
  - Implements a hybrid quantum-classical neural network for license plate character recognition (0-9, A-Z)
  - Provides both a CLI (`ipsem2025-train`) and Python API for model training and testing
  - Supports customizable quantum circuit parameters and training configurations

- **Datasets**: Flexible dataset management system for training models ([Documentation](src/ipsem2025_license_plate/datasets/README.md))
  - Includes multiple dataset types (EMNIST, custom image datasets)
  - Provides a comprehensive CLI tool (`ipsem2025-dataset`) for dataset operations
  - Supports dataset downloading, validation, visualization, and information display
  
- **Utilities**: Various helper tools including logging utilities ([Documentation](src/ipsem2025_license_plate/utils/README.md))
  - Configurable logging system with console and file support
  - Type-safe interfaces and comprehensive error handling

- **License Plate Detection**: YOLOv5-based license plate localization ([Documentation](src/ipsem2025_license_plate/plate_extraction/README.md))
  - Uses a trained YOLOv5 model to detect and extract license plates from images
  - Provides a CLI tool (`ipsem2025-plate-recognizer`) for easy inference on images and datasets


# Development

## Tools and local environment

---

This is a short description of how to work with this project. First create your virtual environment and activate it:

```bash
python -m venv venv
source ./venv/bin/activate
```

or

```bash
poetry shell
```

Install [Poetry](https://python-poetry.org/), its a project management tool, its used during the development to among many things build the package, install and manage dependencies. On the official website there are multiple ways of installing it but the easiest one is to simply install it in your venv with pip:

```bash
pip install poetry
```

Now you can install crucial dependencies. This command will install both package dependencies and development dependencies like `tox` (its similar to a Makefile but for Python), that will also install the package itself in [editable mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html).

```bash
poetry install
```

## Working with tox

---

[Tox](https://tox.wiki/en/latest/) is a generic virtual environment management and test command line tool you can use for:

- checking your package builds and installs correctly under different environments (such as different Python implementations, versions or installation dependencies),

- running your tests in each of the environments with the test tool of choice,

- acting as a frontend to continuous integration servers, greatly reducing boilerplate and merging CI and shell-based testing.

In the `tox.ini` file there a many jobs defined that perform tests, check formatting of the code, format code, lint etc. Their definition can be found but the general ones, that are also checked in the CI are:

- `lint` - Runs PyLint over the code base, both source and tests
- `lint-warn` - Runs PyLint over the code base, both source and tests but checks only for possible warnings, errors and failures, omitting style related concerns
- `check_format` - Checks formatting of the source code
- `format` - Formats the source code with `black`
- `test` - Runs all tests under `tests/`
- `type_checking` - Checks static typing of the source code using `mypy`
- `cov` - Generates and checks test coverage of the source code

Usage of any of those command is very simple, we simply run tox and specify the environment:

```bash
tox -e lint
```

If you want to run the main ones all at once simply run:

```bash
tox
```

Additionally adding the `-p` option will run the commands in parallel.

## Contributing and Releasing

---

The repository follows a semantic release development and release cycle.
This means that all PRs merged into `main`/`master` need to have formats like these:

- `feat(ABC-123): Adds /get api response`
- `fix(MINOR): Fix typo in the CI`
- `fix(#12345): Fix memory leak`
- `ci(Just about anything here): Update Python versions in the CI`

Here is the exact enforced regular expression:

```regex
'^(fix|feat|docs|test|perf|ci|chore)\([^)]+\): .+'
```

Allowed types of conventional commits:

- `fix`: a commit that fixes a bug.
- `feat`: a commit that adds new functionality.
- `docs`: a commit that adds or improves documentation.
- `test`: a commit that adds unit tests.
- `perf`: a commit that improves performance, without functional changes.
- `ci`: a commit that adds or improves the CI configuration.
- `chore`: a catch-all type for any other commits. For instance, if you're implementing a single feature and it makes sense to divide the work into multiple commits, you should mark one commit as feat and the rest as chore.

Releasing the package is done automatically when a commit is merged to `main`/`master`. A new release is created and the `CHANGELOG.md` is updated automatically.

More about the releasing mechanism:
<https://github.com/semantic-release/semantic-release>

# Credits

This package was created with Cookiecutter, and the
`John15321/cookiecutter-poetic-python` project template.

Cookiecutter: <https://github.com/audreyr/cookiecutter>

cookiecutter-poetic-python: <https://github.com/John15321/cookiecutter-poetic-python>
