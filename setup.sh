# creating conda environment with python 3.10.0
conda create --prefix ./venv-team06 python=3.10.0 -y
source activate ./venv-team06/

#  If you do not have poetry installed run the following command and, 
# also by default poetry does not save the virtual environment in 
# your current directory so, to save the .venv in your current directory run the second command.

pip install poetry
poetry config virtualenvs.in-project true


# install the required libraries and package
poetry install

# Activate your poetry environment
poetry shell