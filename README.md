# Prediction Equilibrium for Dynamic Network Flows

## Getting Started

To start working:

* Install Python 3.10 (if not already)
* Install [poetry](https://python-poetry.org/) (if not already)
* Run `poetry config virtualenvs.in-project true`
* Run `poetry install`
* Run `poetry shell` to activate the created virtual environment

### Disable Assertions

Assertions are used extensively throughout the code and therefore slow down the computation significantly.
Often, we use the environment variable `PYTHONOPTIMIZE` to deactivate assert statements.
To disable assertions in your current terminal session, run
 * for Bash: `export PYTHONOPTIMIZE=TRUE`
 * for PowerShell: `$Env:PYTHONOPTIMIZE="TRUE"`
 * for cmd.exe: `set PYTHONOPTIMIZE=TRUE`

### Set PYTHONPATH variable

As the code is split into multiple modules, the `PYTHONPATH` environment variable declares where Python should look for the files.
Use the following command to set the variable:
 * for Bash: `export PYTHONPATH=./src`
 * for PowerShell: `$Env:PYTHONPATH=".\src"`
 * for cmd.exe: `set PYTHONPATH=".\src"`

### Clone the TransportationNetworks repository

Please clone the TransportationNetworks repository from [github.com/bstabler/TransportationNetworks](https://github.com/bstabler/TransportationNetworks) locally.
By default, we assume you cloned the folder to `~/git/TransportationNetworks`, where `~` is your home directory.
However, you can also specify a custom directory by setting the `TNPATH` environment variable.

## Running an experiment

For each experiment described in the manuscript, there is a Python file in the `src/scenarios` folder.
You can run an experiment by simply executing the corresponding file, e.g. `python src/scenarios/sample_scenario.py`.

Upon execution of an experiment, the following steps are executed:

1. Generation of sample flows that use the constant predictor only
2. Generation of training data for the ML predictors (extracting samples of queues and edge loads)
3. Training the ML predictors
4. Evaluating all predictors by using them side by side in 20 simulation runs

As the computation of most steps use only a single core, you can speed up the computation by running the experiment command multiple times in parallel (e.g. number of cores of your processor).
The processes will communicate with each other and only perform tasks that have not been taken by other processes.
All of these steps save their output to the `./out` folder.
Once all steps are executed, the results of step 4 (e.g. tikz plots, visualization files, etc.) are generated into the `out/{experiment}/eval` folder.
You can open files ending in `.vis.json` in a dynamic flow visualization tool available at https://arbeitsgruppetobiasharks.github.io/dynamic-flow-visualization.
