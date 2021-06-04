# Prediction Equilibrium for Dynamic Traffic Assignment

This file descibes how the results in the computational study can be reproduced.
An additional appendix for detailed proofs of the statements in the paper
can be found in [full-paper.pdf](full-paper.pdf).

## Getting Started

To start working:

* Install Python 3.8 (if not already)
* Install pipenv (if not already) `python3 -m pip install --user pipenv`
* Run `pipenv install`
* Run `pipenv shell` to create a shell session with an activated pipenv environment

Assertions are used extensively throughout the code and therefore slow down the computation a lot.
Often, we use the environment variable `PYTHONOPTIMIZE` to deactivate assert statements.
Run the following code, to disable assertions for a bash session:
```
export PYTHONOPTIMIZE=TRUE
```

## Generate Training Data for the Linear Regression Predictor

Generating training data is done in two steps:
* First, flows are generated using an extension based procedure where all commodities have a random inflow rate.
* Then, we take samples of the queue lengths of the generated flows.

To generate flows, run the following command inside an active pipenv shell:
```
python src/main.py generate_flows /path/to/network.arcs /path/to/network.demands /path/to/output-folder
```
This can take several hours, so it is helpful to run multiple processes with the same command.


To take the samples, run
```
python src/main.py take_samples /path/to/output-folder
```

To train the Linear Regression Predictor, samples were taken from the tokyo_small network.
These are included in the Supplementary Material.
After merging all queue lengths into a single file while removing lines with only zeros,
linear regression was applied using the tool [Weka](https://www.cs.waikato.ac.nz/ml/weka/).
The resulting parameters were hardcoded in the corresponding python file.

## Evaluation

### Evaluating the Sample Network

This section shows how to deduce the results for the following sample network:

![Sample Network](data/sample_network.png)

Here, we have a single source s and a single sink t.
We introduce one commodity for each predictor and have the total inflow split equally between them.
This means that we measure, how the average travel times behave
when the different predictors compete in real-time with each other.
As the instance is so small, we can have multiple runs very easily.
Hence, we measure how the average travel times behave when increasing the total network inflow.

To run the experiment, use the following command:
```
python src/main.py evaluate_sample
```
This generates a file `./avg_times_sample.json` with a json array `arr` of the following structure:
* `arr[0]` contains samples for the constant predictor
* `arr[1]` contains samples for the Zero-Predictor
* `arr[2]` contains samples for the Linear Regression Predictor
* `arr[3]` contains samples for the Linear Predictor
* `arr[4]` contains samples for the Regularized Linear Predictor

Furthermore, `arr[i]` is an array of 30/0.25=120 samples with `arr[i][j]` being
the measured average travel time of predictor `i`
in a network with total inflow `j*0.25`.


### Evaluating a Large Network

This section explains how to reproduce the results for the tokyo network.
In this example, we measure the performance of the different predictors for each existing commodity.
More specifically, for each commodity with source-sink-pair `(s, t)`, we introduce 
additional commodities - one for each predictor - with the same source-sink-pair and
a very small network inflow (here: 0.125).

After extending the flow for one focused commodity, the program saves a json file in the
specified output folder.
One such a json file contains besides the id of the observed commodity and other information,
the computed average travel times of the predictors in the order as in the section above. 

In some cases, the travel times of all predictors are the same.
This can happen, if there is only a single (reasonable) path to the sink or if no flow did not arrive
during the observed time horizon.
These entries were removed when creating the boxplot in the paper.

In the paper, the `tokyo_tiny` network was chosen for evaluation.
To run the experiment, use the following command:
```
python src/main.py evaluate_network /path/to/network.arcs /path/to/network.demands /path/to/output-folder
```
