<h1 align="center">Efficient Multi-Model Managemen</h1>
<p align="center">This repository contains the code to our <a href="https://openproceedings.org/2022/conf/edbt/paper-60.pdf"> EDBT '23 short paper<a/>.<p/>

# M<sup>3</sup>lib - Multi Model Management Library

- The code is an extended version of the MMlib (Model Management library)
    - [MMlib paper](https://openproceedings.org/2022/conf/edbt/paper-60.pdf)
    - [MMlib repo](https://github.com/hpides/mmlib)

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
