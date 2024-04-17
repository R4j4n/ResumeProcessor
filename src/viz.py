import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def create_star_graph(keyword_similarity_data):
    """
    Create a star graph from a list of keyword similarities.
    
    Parameters:
    - keyword_similarity_data: A list of lists where each sublist contains a keyword and its similarity score.
    
    Returns:
    - fig: The Matplotlib figure object containing the plot.
    """
    
    keywords, similarity_scores = zip(*keyword_similarity_data)
    similarity_scores = np.array(similarity_scores)

    G = nx.Graph()

    G.add_node("Center")

    for keyword, score in keyword_similarity_data:
        G.add_node(keyword)
        G.add_edge("Center", keyword, weight=score)

    pos = {"Center": (0, 0)}

    circle_pos = nx.circular_layout(keywords)
    for key, val in circle_pos.items():
        pos[key] = (val[0] * similarity_scores[keywords.index(key)], val[1] * similarity_scores[keywords.index(key)])

    fig, ax = plt.subplots(figsize=(12, 12))

    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1)
    nx.draw_networkx_nodes(G, pos, nodelist=["Center"], node_color="skyblue", node_size=3000)
    nx.draw_networkx_labels(G, pos, labels={"Center": "Center"}, font_weight='bold')

    for node, (x, y) in pos.items():
        if node != "Center":
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color="skyblue", node_size=3000)
            plt.text(x, y, s=node, ha='center', va='center', color='black', weight='bold', fontsize=8)

    for node, (x, y) in pos.items():
        if node != "Center":
            score = G.edges[('Center', node)]['weight']
            plt.text((x)/2, (y)/2, s=f"{score:.4f}", ha='center', va='center', color='red', fontsize=10)

    plt.axis('off')

    return fig

