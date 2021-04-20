from os import path
import matsim


def import_from_matsim(output_folder_path):
    net = matsim.read_network(path.join(output_folder_path, "output_network.xml.gz"))
