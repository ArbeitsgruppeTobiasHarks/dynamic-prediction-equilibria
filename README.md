# Dynamic Prediction Equilibrium Flows

## Getting Started

To start working:

* Install Python 3.8 (if not already)
* Install pipenv (if not already) `python3 -m pip install --user pipenv`
* Run `pipenv install`

## Download Network Data

You can download the network data here:

## Generate Training Data

Generating training data is done in two steps:
* First, flows are generated using an extension based procedure where all commodities have a random inflow rate.
* Then, we take samples of the queues of the generated flows.

To generate the flow run the following command inside an active pipenv shell:
```
python predictor/src/main.py generate_flows /path/to/network.arcs /path/to/network.demands ./predictor/out
```

To take the samples, run
```
python predictor/src/main.py take_samples ./predictor/out
```
