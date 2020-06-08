#!/bin/bash
# install system requirements
sudo dnf install -y zlib-devel bzip2 bzip2-devel readline-devel sqlite \
sqlite-devel openssl-devel xz xz-devel libffi-devel make g++

# install pyenv and associated functions
curl https://pyenv.run | bash

# edit .bashrc for pyenv
echo 'export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

source ~/.bashrc
mkdir ~/src
cd ~/src
git clone https://github.com/torralba-lab/im2recipe-Pytorch
cd im2recipe-Pytorch
pyenv install 3.7.7
pyenv virtualenv 3.7.7 food
pyenv local food
pip install -r requirements.txt
pip install torchwordemb