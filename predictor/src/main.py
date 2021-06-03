import os

import click

import importer.matsim_importer
from gnn.DataLoader import generate_queues
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


@click.command(name='import', help="Import a scenario from the matsim output folder")
@click.argument("output_folder", type=click.Path(exists=True, file_okay=False))
def import_matsim(output_folder):
    importer.matsim_importer.import_from_matsim(output_folder)


main.add_command(import_matsim)
main.add_command(generate_flows)
main.add_command(take_samples)

if __name__ == '__main__':
    main()
