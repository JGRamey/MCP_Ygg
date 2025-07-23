# Grok table for improving claude code sessions are below #
Feature,Description,How to Use,Benefits for Coding Sessions,Implementation
Project Memory (CLAUDE.md),"Stores project-specific instructions and context in a CLAUDE.md file, automatically read by Claude Code at the start of each session in the project directory.","Run /init in your project directory to generate CLAUDE.md. Add instructions, coding standards, or project details manually or use # prefix in prompts to append instructions.","Ensures consistent context across sessions, reduces repetitive explanations, and aligns Claude with your project’s coding style. Ideal for team collaboration and long-term projects.","1. Navigate to your project directory: cd /path/to/project.
2. Run claude /init to create CLAUDE.md.
3. Edit CLAUDE.md with a text editor (e.g., nano CLAUDE.md) to add instructions like “Use TypeScript with strict mode” or “Follow REST API conventions.”
4. Append instructions dynamically: claude -p ""# Always use async/await for API calls"". Claude automatically loads CLAUDE.md in subsequent sessions."
Global Memory,Stores consistent behavior settings across all projects in the home directory.,Configure global settings in the home directory’s Claude configuration file (automatically managed by Claude Code).,"Provides a unified development experience, reducing setup time when switching between projects.","1. Locate or create the global config file at ~/.claude/config.json.
2. Edit with a text editor: nano ~/.claude/config.json.
3. Add settings like {""default_language"": ""python"", ""style_guide"": ""PEP8""}.
4. Save and restart Claude Code (claude) to apply globally across projects. No manual command needed for activation; Claude applies settings automatically."
System Memory,"A recursive reference system that shares context across all projects, minimizing redundancy.",Automatically managed by Claude Code; no direct user action required beyond initializing projects.,"Enhances efficiency by reusing relevant context, especially for cross-project tasks.","1. Ensure CLAUDE.md files exist in relevant project directories.
2. Claude automatically builds system memory by indexing CLAUDE.md files and past interactions.
3. To leverage, reference related projects in prompts: claude -p ""Use context from /path/to/other/project for this task"". No explicit activation command; managed in the background."
Slash Commands,"Built-in commands to streamline operations (e.g., /init, /clear, /compact, /hooks).","Type / in the terminal to list available commands. Examples: /init to set up a project, /clear to reset context, /compact to summarize conversation, /hooks to configure hooks.","Automates repetitive tasks, simplifies project management, and improves workflow efficiency by reducing manual input.","1. In the terminal with Claude Code running, type claude / and press Enter to see all commands.
2. Examples:
   - claude /init to initialize a project.
   - claude /clear to reset context window.
   - claude /compact to summarize the session.
   - claude /hooks to open an interactive hook configuration menu.
3. Run commands directly in the Claude Code CLI."
Extended Thinking Mode,"Allows Claude to “self-reflect” and brainstorm steps before coding, ideal for complex tasks.",Toggle with a prompt like “Think hard and create a plan” or enable via specific CLI flags (not always explicitly toggleable; depends on prompt).,"Improves code quality by ensuring thorough planning, reducing errors, and addressing edge cases before implementation.","1. Use a specific prompt: claude -p ""Think step-by-step and create a plan for building a REST API in Flask"". 
2. Claude will outline steps before coding; review and refine with follow-up prompts like “Adjust step 3 to use SQLAlchemy”.
3. No explicit flag; activated via prompt phrasing. Ensure detailed prompts for best results."
One-Shot Mode (-p flag),"Executes a single command or answers one question with high precision, minimizing extra output.","Use claude -p ""specific task"" (e.g., claude -p ""Generate a Python function to parse JSON"").","Ideal for quick, focused tasks like generating snippets or analyzing files, saving time and tokens.","1. Run: claude -p ""Generate a Python function to parse JSON"".
2. Claude processes the prompt and exits after responding, without maintaining session context.
3. Use for single tasks like claude -p ""Debug this error: [paste error]""."
Git Worktrees,"Enables multiple Claude sessions on different project branches simultaneously, using separate worktrees.",Create worktrees: git worktree add ../project-branch branch-name. Run Claude in each worktree.,"Allows parallel task execution, prevents merge conflicts, and boosts productivity for large projects.","1. In your git repo: git worktree add ../feature-branch feature-branch.
2. Navigate to the worktree: cd ../feature-branch.
3. Run Claude: claude or claude /init to set up.
4. Repeat for other branches (e.g., ../bugfix-branch). Run separate Claude instances in each directory."
Git Integration,"Handles git operations like searching history, writing commit messages, resolving conflicts, and creating pull requests.",Prompt Claude with tasks like “Search git history for changes in v1.2.3” or “Write a commit message for my changes.” Use gh CLI for GitHub interactions.,"Streamlines version control, automates commit messages, and simplifies PR creation, saving time and reducing errors.","1. Ensure gh CLI is installed: brew install gh (macOS) or equivalent.
2. Authenticate: gh auth login.
3. Run prompts like:
   - claude -p ""Write a commit message for my changes in src/app.js"".
   - claude -p ""Create a PR for branch feature-x"".
   - claude -p ""Search git history for changes to auth.py in v1.2.3"".
4. Claude executes git commands or uses gh CLI for PRs."
Message Queuing,"Allows sending multiple prompts while Claude is processing, queuing them for the next turn.",Send additional prompts during Claude’s processing; they queue automatically.,"Enhances workflow by letting you plan ahead without waiting, improving session continuity.","1. While Claude processes a prompt, type another: claude -p ""Next, refactor this code"".
2. Claude queues the prompt and processes it after the current task.
3. No explicit activation; built into the CLI. Check for bugs by confirming queued prompts are addressed."
Hooks,"Shell commands triggered at specific points (e.g., PreToolUse, PostToolUse) to automate tasks like formatting or notifications.",Configure via /hooks command or edit JSON config with matcher fields. Example: Auto-format code after edits.,"Automates repetitive tasks, ensures consistency, and integrates with custom workflows.","1. Run claude /hooks to open the interactive hook editor.
2. Or edit ~/.claude/config.json to add hooks like:
   ```json:disable-run"
Model Context Protocol (MCP),"Connects Claude to external tools (e.g., Google Drive, Jira, Slack) or remote MCP servers for additional context.","Set up MCP in project config or use claude -p ""Pull context from Jira"". Requires tool installation.","Expands Claude’s capabilities by integrating with external data sources, improving context for complex tasks.","1. Install required tools (e.g., gh CLI or Jira CLI).
2. Configure MCP in CLAUDE.md or ~/.claude/config.json with API keys or server details.
3. Run: claude -p ""Pull Jira ticket DATA-123 context for this task"".
4. Claude fetches external data if configured. Check Anthropic docs for MCP setup specifics."
Hierarchical Memory Management,"Organizes memory into project, global, and system levels for efficient context storage and retrieval.",Managed automatically; ensure CLAUDE.md is updated for project memory.,"Reduces context loss, improves performance, and ensures relevant information is accessible across sessions.","1. Initialize projects with claude /init to create CLAUDE.md.
2. Update CLAUDE.md with project details.
3. Global memory auto-applies via ~/.claude/config.json.
4. System memory builds automatically; reference other projects with claude -p ""Use context from /path/to/other/project""."
Sub-Agents,Runs multiple Claude instances as sub-agents for parallel task execution.,Launch separate Claude instances in different terminals or worktrees for distinct tasks.,"Increases throughput by delegating tasks simultaneously, mimicking a team of developers.","1. Open multiple terminals.
2. In each, navigate to a project or worktree: cd /path/to/project.
3. Run claude in each terminal with different prompts (e.g., claude -p ""Write tests"" in one, claude -p ""Implement feature"" in another).
4. Manage outputs separately."
Whisper Flow (Voice Input),Enables voice-based input for Claude Code commands.,"Requires setup with compatible terminals (e.g., Warp); specific activation details vary.","Speeds up input for non-typing scenarios, improving accessibility and workflow speed.","1. Install a compatible terminal like Warp.
2. Enable voice input in terminal settings (check Warp docs).
3. Activate Whisper Flow: claude --whisper (if supported) or use terminal’s voice command.
4. Speak prompts like “Generate a Python script.” Confirm availability in Anthropic docs."
IDE Integration,"Integrates with VS Code, JetBrains, and Cursor, showing proposed changes and leveraging open files/LSP diagnostics.",Install Claude Code extension in your IDE; Claude accesses open files and highlights changes.,"Enhances coding by embedding Claude in your IDE, streamlining edits and reducing context-switching.","1. Install Claude Code extension in VS Code (code --install-extension anthropic.claude).
2. Authenticate via extension settings with your Anthropic API key.
3. Open a file in VS Code and run Claude commands via the extension UI or claude -p ""Suggest edits for this file"".
4. Review highlighted changes in the IDE."
Resume Conversations,Picks up past coding conversations without re-explaining context.,Use --resume flag to continue previous sessions.,"Saves time by maintaining session continuity, especially for iterative tasks.","1. Run: claude --resume to load the last session.
2. Claude displays recent context; continue with new prompts like claude -p ""Continue refactoring main.js"".
3. Use --resume <session-id> for specific sessions if multiple exist (check claude /history)."
Plan Mode,"Creates a step-by-step plan before coding, which you can review and refine.",Prompt with “Create a plan for [task]” or use Shift + Tab in some interfaces.,"Ensures alignment on complex tasks, reduces errors, and clarifies implementation steps.","1. Run: claude -p ""Create a plan for a React component"".
2. Claude outputs a numbered plan; refine with claude -p ""Update step 2 to use hooks"".
3. In IDEs, press Shift + Tab (if supported) to trigger plan mode interactively."
Course-Correct on the Fly,Allows interrupting Claude to refine prompts or undo changes.,Press Escape or use the stop button; adjust prompts or request “Undo last change.”,"Maintains control over Claude’s actions, ensuring output aligns with your intent.","1. While Claude processes, press Escape in the terminal or click the stop button in IDEs.
2. Submit a new prompt: claude -p ""Undo last change"" or claude -p ""Refine to use TypeScript"".
3. Claude adjusts based on the new input."
Allowed Tools CLI Flag,Grants session-specific permissions for tools like gh CLI or custom scripts.,"Use --allowedTools flag (e.g., claude --allowedTools gh).",Enhances security and flexibility by controlling tool access per session.,"1. Run: claude --allowedTools gh,prettier ""Task"".
2. Example: claude --allowedTools gh -p ""Create a PR"".
3. Ensure tools are installed (e.g., brew install gh). Claude restricts tool usage to specified ones."
Test-Driven Development (TDD) Support,Generates tests before coding to ensure functionality.,Prompt Claude to “Write tests for [feature]” before implementation.,Improves code reliability and quality by enforcing test coverage from the start.,"1. Run: claude -p ""Write Jest tests for a login function"".
2. Review generated tests in the output file (e.g., tests/login.test.js).
3. Then prompt: claude -p ""Implement login function based on these tests"".
4. Iterate as needed."
Performance Audit and Optimization,Analyzes code for performance issues and implements optimizations.,Prompt with “Run a performance audit on [module] and optimize.” Request before/after metrics.,"Enhances application performance systematically, with measurable results.","1. Run: claude -p ""Run a performance audit on src/server.js and suggest optimizations"".
2. Claude outputs issues (e.g., slow loops) and fixes.
3. Request metrics: claude -p ""Show before/after performance metrics"".
4. Apply changes via Claude’s suggestions or manually."
In-Context Examples,Improves Claude’s responses by providing example Q&A or code snippets.,"Include examples in prompts or CLAUDE.md (e.g., “Follow this format: [example code]”).","Primes Claude for accurate responses, reducing errors in complex tasks.","1. Add to CLAUDE.md: Example: function add(a, b) { return a + b; }.
2. Or include in prompt: claude -p ""Write a function like this: function add(a, b) { return a + b; }"".
3. Claude mimics the style/format in responses."

# 🚀 Claude Code Session Usage Optimization Guide

## 📋 Quick Reference for Efficient Sessions

### **🔧 Essential Setup (One-Time)**
1. **Enable Memory Server** - Already configured in `claude.md`
2. **Use Condensed Context** - Reference `CLAUDE_SESSION.md` instead of full `CLAUDE.md`
3. **Verify MCP Servers** - Ensure memory and sequential-thinking servers are active

### **💡 Session Start Best Practices**

#### Instead of:
```
"Hey Claude, can you help me with my project?"
```

#### Use:
```
"Use CLAUDE_SESSION.md for context. I need help with [specific task]."
```

### **🎯 Task-Specific Optimization**

#### **File Operations**
- ✅ **DO**: `Read /path/to/specific/file.py`
- ❌ **DON'T**: `Show me all Python files and their contents`

#### **Search Operations**
- ✅ **DO**: `Grep "specific_function" --include="*.py"`
- ❌ **DON'T**: `Find all functions in the codebase`

#### **Code Changes**
- ✅ **DO**: `Edit specific_file.py - replace line 25`
- ❌ **DON'T**: `Review all files and suggest improvements`

### **📊 Tool Usage Efficiency**

#### **Batch Operations**
```bash
# Efficient: Single message with multiple tools
Bash: git status
Bash: git diff
Bash: make lint

# Inefficient: Separate messages for each command
```

#### **Targeted Searches**
- Use `Task` tool for complex/unknown file searches
- Use `Glob` for known file patterns
- Use `Grep` for specific content patterns

### **🔄 Session Continuation**

#### **Optimal Session Start:**
1. "Use CLAUDE_SESSION.md for context"
2. "I'm working on [specific component]"
3. "My goal is [specific outcome]"

#### **Avoid:**
- Reading entire chat logs
- Reviewing all project files
- General "what should I do next?" questions

### **📁 File Management**

#### **Reference Files (Use These):**
- `CLAUDE_SESSION.md` - Current project status
- `Makefile` - Available commands
- `requirements.txt` - Dependencies
- `docker-compose.yml` - Service configuration

#### **Avoid Reading Unless Necessary:**
- `CLAUDE.md` (3000+ lines - use SESSION version)
- Complete chat logs
- Large CSV files (use specific paths)
- Generated output files

### **⚡ Quick Commands Reference**

```bash
# System Status
streamlit run main_dashboard.py --server.port 8502  # Main interface
make lint                                           # Code quality
make test                                           # Run tests
make docker                                         # Start services

# Health Checks
curl http://localhost:8502                          # Streamlit
curl http://localhost:8000/health                   # API
curl http://localhost:7474                          # Neo4j
```

### **🎯 Common Task Patterns**

#### **Adding New Features**
1. "I want to add [feature] to [specific module]"
2. Read only the target file
3. Make specific changes
4. Test the specific component

#### **Debugging Issues**
1. "I'm getting [specific error] in [specific file]"
2. Read the error-producing file
3. Check related configuration
4. Apply targeted fix

#### **Code Review**
1. "Review [specific file] for [specific issue]"
2. Focus on the problem area
3. Suggest targeted improvements

### **📈 Session Efficiency Metrics**

#### **High Efficiency Indicators:**
- 🟢 Specific file paths mentioned
- 🟢 Clear task definition
- 🟢 Limited scope requests
- 🟢 Batched tool operations

#### **Low Efficiency Indicators:**
- 🔴 "Show me everything" requests
- 🔴 Multiple file exploration
- 🔴 Vague task descriptions
- 🔴 Sequential tool calls

### **🛠️ Troubleshooting Session Issues**

#### **If Sessions Deplete Quickly:**
1. Check if memory server is enabled
2. Use `CLAUDE_SESSION.md` for context
3. Be more specific in requests
4. Avoid broad exploration tasks

#### **If Memory Issues Occur:**
1. Clear memory: `rm /Users/grant/Documents/GitHub/MCP_Ygg/chat_logs/memory.json`
2. Restart Claude Code
3. Use condensed context files

### **🎉 Pro Tips**

1. **Start Focused** - Know what you want to accomplish
2. **Use Specific Paths** - Avoid directory exploration
3. **Batch Operations** - Group related tasks
4. **Reference Documentation** - Use existing guides
5. **Test Incrementally** - Small changes, frequent testing

### **📋 Session Template**

```
Use CLAUDE_SESSION.md for context.

Task: [Specific goal]
Files: [Specific files to modify]  
Scope: [Limited scope description]

Please [specific action] in [specific location].
```

---

**💡 Remember:** Efficient sessions = Specific requests + Targeted actions + Minimal context loading