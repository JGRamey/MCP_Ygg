Use context7 and follow these guidelines to ensure modular, maintainable, and concise code:

1. **Modular Structure**: Split the code into multiple files/modules, each with a single responsibility (e.g., configuration, data models, core logic, utilities, CLI, visualization). For example:
   - config.py: Configuration settings
   - models.py: Data classes or schemas
   - core.py: Main business logic
   - utils.py: Utility functions
   - cli.py: Command-line interface (if applicable)
   Ensure each module is focused and avoid creating monolithic coding files

2. **Clear Responsibilities**: Each module should handle specific and cohesive functions. Use clear, descriptive names for files, classes, and functions.

3. **Conciseness**: Keep code concise and avoid unnecessary features. Only include functionality explicitly requested or essential for the agent/features.

4. **Best Practices**:
   - Include proper error handling and logging.
   - Use type hints for function signatures and variables (if applicable in the language).
   - Follow PEP 8 for Python
   - Write clear, concise comments for each module, class, and function.
   - Use meaningful variable and function names.

5. **Imports**: Use explicit imports between modules. Avoid circular imports and ensure dependencies are clear.

6. **No Monolithic Files**: Do not combine all functionality into a single file. If the task is complex, propose a directory structure and explain the purpose of each module.

7. **Specific Requirements**: Implement [list specific features, e.g., 'functions for PageRank and betweenness centrality', 'a REST endpoint for user authentication']. Exclude unrelated features like [e.g., 'visualization unless specified', 'database migrations unless requested'].

8. **Testing and Validation**: If applicable, include a small example or test case in the `tests/` foler in the directory to demonstrate usage.

9. **Output**: Provide the code as separate files (within the same folder if applicable) with clear filenames. For each file, include:
   - A brief description of its purpose.
   - The complete code with proper imports.
   - A note on how it integrates with other modules.