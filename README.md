# COMP 472 Mini-Project 1
## Machine Learning experiments

Student ID 40058095

## Stucture
```
/ (root dir)
data/..............(data here)
....BBC/
....drug200.csv
docs/..............(discussions and presentation slides)
....bbc-discussion.txt
....drug-discussion.txt
....MP1 results & observations.pdf
out/...............(output files)
...BBC-distribution.pdf
...bbc_performance.txt
...drug-distribution.pdf
...drugs_performance.txt
README.md (this file)
requirements.txt 
task1.py
task2.py
```
## Dependencies
- [matplotlib](https://matplotlib.org/stable/users/installing.html)
- [scikit-learn](https://scikit-learn.org/stable/)

Both include many more libraries (see `requirements.txt`)


## Getting Started
### Create a virtual env e.g.
```
virtualenv -p python3 venv
source venv/bin/activate
```

### Install dependencies
With the virtualenv activated:
```
pip install -r requirements.txt
```

### Run programs
```
python task1.py
python task2.py
```
