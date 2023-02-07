# M<sup>3</sup>lib - Multi Model Management Library

- This is the code for our EDBT short paper:
    - Reference will follow once oficially published
- The code is an extended version of the MMlib (Model Management library)
    - you can find the repo of the plain MMlib -- [here](https://github.com/hpides/mmlib)

## Installation

### Option 1: Docker

- **Requirements**: Docker installed
- **Build Library**
    - clone this repo
    - run the script `generate-archives-docker.sh`
        - it runs a docker container and builds the *mmlib* in it
        - the created `dist` directory is copied back to repository root
        - it contains the `.whl` file that can be used to install the library with pip (see below)
- **Install**
    - to install mmlib run: `pip install <PATH>/dist/mmlib-0.0.1-py3-none-any.whl`

### Option 2: Local Build

- **Requirements**: Python 3.8 and Python `venv`
- **Build Library**
    - run the script `generate-archives.sh`
        - it creates a virtual environment, activates it, and installs all requirements
        - afterward it builds the library, and a `dist` directory containing the `.whl` file is created
- **Install**
    - to install mmlib run: `pip install <PATH>/dist/mmlib-0.0.1-py3-none-any.whl`
