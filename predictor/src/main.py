import click

import importer.matsim_importer


@click.group()
def main():
    pass


@click.command(name='import', help="Import a scenario from the matsim output folder")
@click.argument("output_folder", type=click.Path(exists=True, file_okay=False))
def import_matsim(output_folder):
    importer.matsim_importer.import_from_matsim(output_folder)


@click.command()
def stop():
    print("running command `stop`")


main.add_command(import_matsim)
main.add_command(stop)

if __name__ == '__main__':
    main()
