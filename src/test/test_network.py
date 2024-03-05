from test.sample_network import build_sample_network


def test_build():
    network = build_sample_network()
    assert len(network.graph.nodes) == 4
    assert len(network.graph.edges) == 5
    assert len(network.graph.nodes[0].outgoing_edges) == 2
    assert len(network.graph.nodes[2].incoming_edges) == 2
