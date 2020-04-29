# Distributed Bayesian Optimization for Multi-Agent Systems

This package is developed for Python 3.7.

# Setup instructions for Debian-like environments

1. (Optional) Set up and activate a virtual environment
```
virtualenv -p python3 ~/.venvs/dboenv
source ~/.venvs/dboenv/bin/activate
```

2. (Optional) Set up alias for environment
```
echo 'alias dboenv="source ~/.venvs/dboenv/bin/activate"' >> ~/.zshrc
source ~/.zshrc  
dboenv      # activate
deactivate  # deactivate
```

2. Install python dependencies and dbo
```
git clone git@github.com:FilipKlaesson/dbo.git
cd dbo
pip install -r requirements.txt
python setup.py install
```
