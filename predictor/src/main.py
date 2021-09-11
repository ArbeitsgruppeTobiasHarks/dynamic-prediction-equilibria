import click

import eval.evaluate_network
import eval.evaluate_sample
import sioux_falls.sioux_falls_scenario
import tokyo.tokyo_scenario


@click.group()
def main():
    pass


@click.command(name='evaluate_sample', help="Evaluate the sample graph using demands in (0, 30)")
def evaluate_sample():
    eval.evaluate_sample.eval_sample()


@click.command(name='run_tokyo',
               help="Run the Tokyo Scenario: Build training data, train and evaluate.")
@click.argument("arcs_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("demands_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_folder", type=click.Path(exists=True, file_okay=False))
def run_tokyo_scenario(arcs_path: str, demands_path: str, output_folder: str):
    tokyo.tokyo_scenario.run_scenario(arcs_path, demands_path, output_folder)


@click.command(name='run_sioux_falls',
               help="Run the Sioux Falls Scenario: Build training data, train and evaluate.")
@click.argument("tntp_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_folder", type=click.Path(exists=True, file_okay=False))
def run_sioux_falls_scenario(tntp_path: str, output_folder: str):
    sioux_falls.sioux_falls_scenario.run_scenario(tntp_path, output_folder)


main.add_command(evaluate_sample)
main.add_command(run_sioux_falls_scenario)
main.add_command(run_tokyo_scenario)

if __name__ == '__main__':
    main()
