import os

import click

import eval.evaluate_sample
import eval.evaluate_network
from importer.generate_queues import generate_queues
from importer.build_test_flows import build_flows_from_demand


@click.group()
def main():
    pass


@click.command(name='generate_flows', help="Generate flows using the constant predictor and random inflow rates.")
@click.argument("network_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("commodities_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_folder", type=click.Path(exists=True, file_okay=False))
def generate_flows(network_path: str, commodities_path: str, output_folder: str):
    print("Generating 200 Flows with random inflow between 20 and 100.")
    print("This might use up storage of ~22 GB")
    build_flows_from_demand(network_path, commodities_path, os.path.join(output_folder, "flows"), 200)


@click.command(name='take_samples', help="Generate training data from generated flows.")
@click.argument("output_folder", type=click.Path(exists=True, file_okay=False))
def take_samples(output_folder: str):
    generate_queues(5, 5, os.path.join(output_folder, "flows"), os.path.join(output_folder, "queues"))


@click.command(name='evaluate_sample', help="Evaluate the sample graph using demands in (0, 30)")
def evaluate_sample():
    eval.evaluate_sample.eval_sample()


@click.command(name='evaluate_network', help="Evaluate a network from file.")
@click.argument("network_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("commodities_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_folder", type=click.Path(exists=True, file_okay=False))
def evaluate_network(network_path: str, commodities_path: str, output_folder: str):
    eval.evaluate_network.eval_network(network_path, commodities_path, output_folder)


main.add_command(generate_flows)
main.add_command(take_samples)
main.add_command(evaluate_sample)
main.add_command(evaluate_network)


if __name__ == '__main__':
    main()
