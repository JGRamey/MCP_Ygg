# AI Claim Analyzer Agent for MCP Server

## Overview
This document outlines the requirements and guidelines for developing an AI agent for a Minecraft (MCP) server. The agent’s primary function is to analyze claims from various sources (books, texts, video transcripts, debates, etc.), search for similar claims, and fact-check their accuracy. The agent should integrate seamlessly with the MCP server environment, providing real-time or near-real-time feedback to players or administrators.

## Objectives
- **Claim Analysis**: Extract and interpret claims from diverse input formats (text, transcripts, etc.).
- **Claim Search**: Identify similar claims across external sources or a predefined database.
- **Fact-Checking**: Evaluate the accuracy of claims using reliable sources and reasoning.
- **MCP Integration**: Operate within the Minecraft server environment, interacting with players via chat, commands, or in-game interfaces.
- **User-Friendly Output**: Provide clear, concise, and actionable fact-checking results to players or server admins.

## Functional Requirements

### 1. Input Processing
- **Supported Formats**:
  - Plain text (e.g., books, chat messages, or server logs).
  - Video transcripts (text-based, e.g., SRT or plain text files).
  - Debate transcripts or structured text from external sources.
- **Input Sources**:
  - In-game sources: Minecraft books, signs, or player chat.
  - External uploads: Allow server admins to upload text files or URLs containing transcripts or texts.
  - Real-time input: Process claims made in player chats or commands.
- **Claim Extraction**:
  - Use natural language processing (NLP) to identify claims (assertions or statements that can be verified) in the input.
  - Example: In the text “The Earth is flat,” identify “The Earth is flat” as a claim.
  - Handle ambiguous or complex claims by breaking them into verifiable components.

### 2. Claim Search
- **Search Mechanism**:
  - Query a database of known claims or use an external API (e.g., web search or fact-checking services like Snopes, PolitiFact, or Google Fact Check Tools).
  - Compare input claims to similar claims using semantic similarity (e.g., cosine similarity with embeddings like BERT or SentenceTransformers).
- **Sources**:
  - Prioritize credible, authoritative sources (e.g., peer-reviewed journals, reputable news outlets, government reports).
  - Allow admins to configure a custom database of trusted sources or claims for the server.
- **Caching**:
  - Cache search results locally to reduce API calls and improve performance on the MCP server.

### 3. Fact-Checking
- **Evaluation Criteria**:
  - Verify claims against credible sources.
  - Assign a confidence score or label (e.g., True, False, Partially True, Unverified) based on evidence.
  - Handle conflicting sources by weighing their credibility (e.g., primary sources > secondary sources).
- **Reasoning**:
  - Use logical reasoning to assess claims, especially for subjective or nuanced statements.
  - Provide explanations for fact-checking results, citing sources and reasoning steps.
- **Edge Cases**:
  - Handle unprovable claims (e.g., opinions, predictions) by labeling them as “Unverifiable” with an explanation.
  - Detect and flag satirical or intentionally false claims.

### 4. MCP Server Integration
- **Platform**:
  - Build the agent as a Minecraft plugin (e.g., using Spigot/Bukkit for MCP servers) or a standalone Python/Node.js script interfacing with the server via RCON or a custom API.
- **Interaction Methods**:
  - **Chat Commands**: Allow players to submit claims via commands (e.g., `/factcheck <claim>`).
  - **In-Game Books**: Analyze claims in written books when submitted to the agent (e.g., via a special NPC or chest).
  - **Chat Monitoring**: Optionally monitor public chat for claims and provide real-time feedback (configurable by admins).
  - **GUI Interface**: Create a simple in-game GUI (e.g., using Minecraft’s inventory menus) to display results.
- **Performance**:
  - Optimize for low latency to avoid server lag.
  - Process claims asynchronously to prevent blocking the main server thread.
- **Permissions**:
  - Restrict fact-checking features to specific player roles (e.g., admins, moderators) or make them available to all players.
  - Allow admins to toggle features (e.g., chat monitoring) via a config file.

### 5. Output
- **Format**:
  - Return results in-game via chat messages, books, or GUI.
  - Example output: “Claim: ‘The Earth is flat.’ Verdict: False. Evidence: NASA satellite imagery and peer-reviewed studies confirm the Earth is an oblate spheroid.”
- **Detail Levels**:
  - Provide a brief verdict (e.g., “False”) for quick feedback.
  - Offer detailed explanations with sources on request (e.g., via `/factcheck details`).
- **Logging**:
  - Log all fact-checking results to a server file or database for admin review.
  - Include timestamps, player names, claims, and verdicts.

## Technical Implementation

### Tech Stack
- **Core Logic**:
  - Use Python for NLP and fact-checking logic due to its robust libraries (e.g., SpaCy, Transformers, NLTK).
  - Alternatively, use Node.js for better integration with web APIs if real-time web scraping is needed.
- **NLP Libraries**:
  - SpaCy or NLTK for claim extraction.
  - SentenceTransformers or Hugging Face models for semantic similarity to find similar claims.
- **Fact-Checking APIs**:
  - Integrate with APIs like Google Fact Check Tools, MediaWiki (for Wikipedia), or custom-built claim databases.
- **MCP Integration**:
  - Use a Spigot/Bukkit plugin to handle in-game interactions.
  - Use RCON or a WebSocket-based API for external script communication with the server.
- **Database**:
  - SQLite or MySQL for caching claims, search results, and fact-checking outcomes.
  - Store source credibility rankings for efficient fact-checking.

### Development Steps
1. **Setup Environment**:
   - Install necessary libraries (e.g., SpaCy, Transformers, requests for API calls).
   - Set up a Spigot/Bukkit development environment for MCP plugin development.
2. **Claim Extraction**:
   - Implement an NLP pipeline to parse input and identify claims.
   - Test with sample inputs (e.g., Minecraft books, chat messages, debate transcripts).
3. **Search and Fact-Checking**:
   - Develop a module to query external APIs or a local database for similar claims.
   - Implement a fact-checking algorithm that cross-references claims with credible sources.
4. **MCP Integration**:
   - Create a plugin with commands (e.g., `/factcheck`) and event listeners for chat or book submissions.
   - Test integration with a local MCP server to ensure stability.
5. **Testing and Optimization**:
   - Test with diverse claims (factual, false, subjective) to ensure robustness.
   - Optimize for performance to minimize server impact.
6. **Documentation**:
   - Provide a user guide for players and admins on how to use the agent.
   - Include configuration instructions for enabling/disabling features.

## Configuration
- Create a `config.yml` file for the plugin/script with options:
  - `api_keys`: Store API keys for external fact-checking services.
  - `trusted_sources`: List of URLs or domains for credible sources.
  - `chat_monitoring`: Enable/disable real-time chat claim analysis.
  - `max_results`: Limit the number of sources returned per fact-check.
  - `language`: Set the language for NLP processing (default: English).

## Example Usage
- **Player Command**:
  ```
  /factcheck The moon landing was faked
  ```
  **Response**:
  ```
  Claim: The moon landing was faked.
  Verdict: False.
  Evidence: NASA’s Apollo program provided extensive evidence, including lunar rocks and photos, verified by independent scientists.
  ```
- **Book Submission**:
  - Player places a written book in a designated chest.
  - Agent extracts claims, fact-checks them, and returns results in a new book or chat message.

## Constraints
- **Performance**: Ensure the agent does not overload the MCP server. Use asynchronous processing for API calls and NLP tasks.
- **API Limits**: Handle rate limits for external APIs by caching results and implementing retry logic.
- **Privacy**: Avoid storing sensitive player data unless explicitly required by admins.
- **Language Support**: Initially focus on English; add multilingual support if needed.

## Future Enhancements
- Add support for real-time video transcript analysis via speech-to-text APIs.
- Implement a machine learning model to improve claim detection accuracy over time.
- Create a web dashboard for admins to review fact-checking logs and manage sources.

## Deliverables
- A Spigot/Bukkit plugin or external script for the AI agent.
- A configuration file with customizable settings.
- A user guide for players and admins.
- Sample test cases and expected outputs.
