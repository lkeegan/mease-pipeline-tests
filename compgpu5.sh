# remove & clone mease-lab-to-nwb
rm -rf mease-lab-to-nwb
git clone https://github.com/lkeegan/mease-lab-to-nwb
cd mease-lab-to-nwb
git checkout add_ci

# remove & recreate conda env
conda remove --name measelab --all -y
conda env create -f mease-env.yml
conda activate measelab

# install
python setup.py develop
