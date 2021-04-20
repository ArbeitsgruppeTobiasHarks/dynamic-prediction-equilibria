# Predicted Dynamic Flows

## Predictor
To start working with the predictor:
* Install Python 3.8 (if not already)
* Install pipenv (if not already) `python3 -m pip install --user pipenv`
* Run `pipenv install`

### Import from Matsim
To start the import procedure from a simulated matsim scenario,
use the following command inside the activated pipenv:
```sh
python predictor/src/main.py import path/to/matsim/output
```