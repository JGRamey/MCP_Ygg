#!/usr/bin/env python3
"""
Chat Logger for MCP Yggdrasil
Automatically logs conversations between JGR and Claude with timestamps and action summaries
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class ChatLogger:
    def __init__(self, base_path: str = "/Users/grant/Documents/GitHub/MCP_Ygg"):
        self.base_path = Path(base_path)
        self.chat_logs_dir = self.base_path / "chat_logs"
        self.chat_logs_dir.mkdir(exist_ok=True)
        
        # Generate session filename based on current date/time
        now = datetime.now()
        self.session_file = f"{now.strftime('%Y-%m-%d_%H-%M')}.md"
        self.log_path = self.chat_logs_dir / self.session_file
        
        # Initialize session if not exists
        if not self.log_path.exists():
            self.initialize_session()
    
    def initialize_session(self):
        """Initialize a new chat session log"""
        now = datetime.now()
        header = f"""# MCP Yggdrasil Chat Log
**Session Date:** {now.strftime('%Y-%m-%d')}  
**Session Time:** {now.strftime('%H:%M')}  
**Participants:** JGR (User), Claude (Assistant)

---

## Session Summary
*[Session summary will be updated as conversation progresses]*

---

## Chat Log

"""
        with open(self.log_path, 'w') as f:
            f.write(header)
    
    def log_message(self, speaker: str, message: str, actions: Optional[List[str]] = None):
        """
        Log a message from either JGR or Claude
        
        Args:
            speaker: "JGR" or "Claude"
            message: The actual message content
            actions: List of actions taken (for Claude messages)
        """
        timestamp = datetime.now().strftime('[%H:%M:%S]')
        
        log_entry = f"\n### {timestamp} {speaker}:\n"
        
        # Add action summaries for Claude messages
        if actions and speaker == "Claude":
            for action in actions:
                log_entry += f"*ACTION: {action}*\n\n"
        
        log_entry += f"{message}\n"
        
        # Append to log file
        with open(self.log_path, 'a') as f:
            f.write(log_entry)
    
    def log_user_message(self, message: str):
        """Log a message from JGR (User)"""
        self.log_message("JGR", message)
    
    def log_claude_message(self, message: str, actions: Optional[List[str]] = None):
        """Log a message from Claude with optional actions"""
        self.log_message("Claude", message, actions)
    
    def update_session_summary(self, summary: str):
        """Update the session summary at the top of the log"""
        # Read current content
        with open(self.log_path, 'r') as f:
            content = f.read()
        
        # Replace the summary placeholder
        updated_content = content.replace(
            "*[Session summary will be updated as conversation progresses]*",
            summary
        )
        
        # Write back to file
        with open(self.log_path, 'w') as f:
            f.write(updated_content)
    
    def get_current_session_file(self) -> str:
        """Return the current session filename"""
        return str(self.log_path)
    
    def list_all_sessions(self) -> List[str]:
        """List all chat log sessions"""
        return [f.name for f in self.chat_logs_dir.glob("*.md")]

# Example usage and testing
if __name__ == "__main__":
    logger = ChatLogger()
    
    # Test logging
    logger.log_user_message("This is a test user message")
    logger.log_claude_message(
        "This is a test Claude response", 
        actions=["Created test file", "Updated documentation"]
    )
    
    logger.update_session_summary("Test session for chat logging functionality")
    
    print(f"Log created at: {logger.get_current_session_file()}")
    print(f"All sessions: {logger.list_all_sessions()}")