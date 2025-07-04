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