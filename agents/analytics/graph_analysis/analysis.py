async def _analyze_structure(graph: nx.Graph) -> NetworkAnalysisResult:
    """Analyze overall network structure."""

    logging.info("Analyzing network structure...")

    # Basic structural metrics
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    density = nx.density(graph)

    # Connectivity analysis
    is_connected = nx.is_connected(graph)
    num_components = nx.number_connected_components(graph)

    if is_connected:
        diameter = nx.diameter(graph)
        avg_path_length = nx.average_shortest_path_length(graph)
    else:
        # For disconnected graphs, calculate for largest component
        largest_cc = max(nx.connected_components(graph), key=len)
        largest_subgraph = graph.subgraph(largest_cc)
        diameter = nx.diameter(largest_subgraph) if len(largest_cc) > 1 else 0
        avg_path_length = (
            nx.average_shortest_path_length(largest_subgraph)
            if len(largest_cc) > 1
            else 0
        )

    # Clustering metrics
    avg_clustering = nx.average_clustering(graph)
    transitivity = nx.transitivity(graph)

    # Degree distribution
    degrees = [d for n, d in graph.degree()]
    avg_degree = np.mean(degrees)
    degree_std = np.std(degrees)
    max_degree = max(degrees) if degrees else 0

    # Small world properties
    # Compare clustering and path length to random graph
    try:
        random_graph = nx.erdos_renyi_graph(num_nodes, density)
        random_clustering = nx.average_clustering(random_graph)
        random_path_length = (
            nx.average_shortest_path_length(random_graph)
            if nx.is_connected(random_graph)
            else 0
        )

        clustering_ratio = (
            avg_clustering / random_clustering if random_clustering > 0 else 0
        )
        path_length_ratio = (
            avg_path_length / random_path_length if random_path_length > 0 else 0
        )

        # Small world coefficient
        small_world_coeff = (
            clustering_ratio / path_length_ratio if path_length_ratio > 0 else 0
        )
    except Exception:
        clustering_ratio = 0
        path_length_ratio = 0
        small_world_coeff = 0

    # Assortativity (degree correlation)
    try:
        assortativity = nx.degree_assortativity_coefficient(graph)
    except Exception:
        assortativity = 0

    # Create basic node metrics
    node_metrics = []
    clustering_dict = nx.clustering(graph)

    for node in graph.nodes:
        node_data = graph.nodes[node]

        metrics = NodeMetrics(
            node_id=node,
            centrality_scores={
                "degree": graph.degree(node),
                "normalized_degree": (
                    graph.degree(node) / (num_nodes - 1) if num_nodes > 1 else 0
                ),
            },
            community_id=None,
            clustering_coefficient=clustering_dict.get(node, 0),
            degree=graph.degree(node),
            metadata=node_data,
        )
        node_metrics.append(metrics)

    # Compile graph metrics
    graph_metrics = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": density,
        "is_connected": is_connected,
        "num_components": num_components,
        "diameter": diameter,
        "avg_path_length": avg_path_length,
        "avg_clustering": avg_clustering,
        "transitivity": transitivity,
        "avg_degree": avg_degree,
        "degree_std": degree_std,
        "max_degree": max_degree,
        "assortativity": assortativity,
        "small_world_coeff": small_world_coeff,
        "clustering_ratio": clustering_ratio,
        "path_length_ratio": path_length_ratio,
    }

    # Generate insights
    insights = [
        f"Network has {num_nodes} nodes and {num_edges} edges",
        f"Network density: {density:.4f}",
        f"Average degree: {avg_degree:.2f}",
        f"Average clustering coefficient: {avg_clustering:.4f}",
        f"Network transitivity: {transitivity:.4f}",
    ]

    if is_connected:
        insights.append(f"Network is connected with diameter {diameter}")
        insights.append(f"Average shortest path length: {avg_path_length:.2f}")
    else:
        insights.append(f"Network has {num_components} connected components")

    # Small world analysis
    if small_world_coeff > 1:
        insights.append(
            f"Network exhibits small-world properties (coefficient: {small_world_coeff:.2f})"
        )

    # Assortativity analysis
    if assortativity > 0.1:
        insights.append("Network shows assortative mixing (similar nodes connect)")
    elif assortativity < -0.1:
        insights.append(
            "Network shows disassortative mixing (dissimilar nodes connect)"
        )
    else:
        insights.append("Network shows neutral mixing patterns")

    recommendations = [
        "Monitor network density changes to track growth patterns",
        "Consider the small-world properties for information diffusion strategies",
        "Use clustering metrics to identify cohesive subgroups",
        "Leverage assortativity patterns for targeted interventions",
    ]

    return NetworkAnalysisResult(
        analysis_type=AnalysisType.STRUCTURAL_ANALYSIS,
        graph_metrics=graph_metrics,
        node_metrics=node_metrics,
        communities=[],
        insights=insights,
        recommendations=recommendations,
        generated_at=datetime.now(timezone.utc),
        execution_time=0.0,
    )
