from os import listdir
from os.path import isfile, join
import matsim


def import_from_matsim(output_folder_path):
    files = [f for f in listdir(output_folder_path) if isfile(join(output_folder_path, f))]
    net_file = [f for f in files if f.endswith("output_network.xml.gz")][0]
    print("Reading network... This may take a while.")
    net = matsim.read_network(join(output_folder_path, net_file))
    print("Successfully read network ({0} nodes, {1} links)".format(len(net.nodes), len(net.links)))
    events_file = [f for f in files if f.endswith("output_events.xml")][0]
    events_reader = matsim.event_reader(join(output_folder_path, events_file), types=['linkEnter', 'linkLeave'])

    def handle_event(event):
        print(event)

    first_event = next(events_reader)
    first_time = first_event.time
    last_time = first_time + 10
    handle_event(first_event)

    for event in events_reader:
        if event.time > last_time:
            break
        handle_event(event)
