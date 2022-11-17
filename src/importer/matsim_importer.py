from os import listdir
from os.path import isfile, join
import matsim


def import_from_matsim(output_folder_path):
    files = [f for f in listdir(output_folder_path) if isfile(join(output_folder_path, f))]
    net_file = [f for f in files if f.endswith("output_network.xml.gz")][0]
    print("Reading network... This may take a while.")
    net = matsim.read_network(join(output_folder_path, net_file))
    print("Successfully read network ({0} nodes, {1} links)".format(len(net.nodes), len(net.links)))
    events_file = [f for f in files if f.endswith("output_events.xml.gz")][0]
    events_reader = matsim.event_reader(join(output_folder_path, events_file), types='left link,entered link')

    first_time = 0
    last_time = first_time + 1000

    for event in events_reader:
        if event["time"] > last_time:
            break
        print(event)
