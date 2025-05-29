import networkx as nx
import matplotlib.pyplot as plt

def add_nodes_edges(graph, node, parent=None):
    graph.add_node(id(node), label=str(node.board)) #round(node.reward,3)
    if parent is not None:
        graph.add_edge(id(parent), id(node))
    for child in node.children:
        add_nodes_edges(graph, child, node)

def draw_tree(root):
    G = nx.DiGraph()
    add_nodes_edges(G, root)

    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    
    nx.draw(G, pos, with_labels=False, arrows=True, node_size=1500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, labels)
    plt.show()