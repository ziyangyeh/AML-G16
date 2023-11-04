#! /bin/bash

git submodule update --init --recursive
git submodule update --remote --merge

# install cpp extensions, the pointnet++ library
cd openpoints/cpp/pointnet2_batch
python setup.py install
cd ../

# grid_subsampling library. necessary only if interested in S3DIS_sphere
cd subsampling
python setup.py build_ext --inplace
cd ..


# point transformer library. Necessary only if interested in Point Transformer and Stratified Transformer
# cd pointops/
# python setup.py install
# cd ..

# Blow are functions that optional. Necessary only if interested in reconstruction tasks such as completion
cd chamfer_dist
python setup.py install --user
cd ../emd
python setup.py install --user
cd ../../../

sed -i "1 a from . import *" openpoints/__init__.py

# PTv1 & PTv2 or precise eval
cd Pointcept/libs/pointops
python setup.py install --user
cd ../../..