# wip recipe to install everything from scratch on compgpu5

## mease-lab-to-nwb

- clone repo
  - `git clone https://github.com/catalystneuro/mease-lab-to-nwb`
  - `cd mease-lab-to-nwb`
- install deps in conda environment (note: mostly not pinned versions)
  - `conda env create -f mease-env.yml`
  - `conda activate measelab`
- install mease-lab-to-nwb
  - `python setup.py develop`
