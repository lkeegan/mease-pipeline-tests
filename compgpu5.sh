# run in root of mease-lab-to-nwb repo

# remove env
conda remove --name measelab --all

# create env
conda env create -f mease-env.yml

# activate env
conda activate measelab

# install
python setup.py develop
