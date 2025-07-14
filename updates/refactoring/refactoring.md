# Refactoring Plan: `network_analyzer.py`

## 1. What is being refactored?

The file `/Users/grant/Documents/GitHub/MCP_Ygg/agents/analytics/graph_analysis/network_analyzer.py` is being refactored. This is a large, monolithic file containing data models, configuration, and the main analysis logic.

## 2. Why is it being refactored?

The file is being refactored to improve modularity, maintainability, and readability. By breaking the file down into smaller, more focused modules, the code will be easier to understand, test, and extend.

## 3. How will it be refactored?

The refactoring will be done in the following steps:

1.  **Extract Data Models**: The `NodeMetrics`, `CommunityInfo`, and `NetworkAnalysisResult` data classes, along with the `AnalysisType`, `CentralityMeasure`, and `CommunityAlgorithm` enums, will be moved to a new file: `agents/analytics/graph_analysis/models.py`.
2.  **Extract Configuration**: The `NetworkConfig` class will be moved to a new file: `agents/analytics/graph_analysis/config.py`.
3.  **Extract Analysis Logic**: The individual analysis functions (e.g., `_analyze_centrality`, `_analyze_communities`) will be moved to a new file: `agents/analytics/graph_analysis/analysis.py`.
4.  **Simplify `NetworkAnalyzer`**: The main `NetworkAnalyzer` class will be simplified to be an orchestrator that calls the functions in the other modules. It will remain in the `network_analyzer.py` file.
