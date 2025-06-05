from IPython.display import display, HTML, Markdown, clear_output
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
import json
import ipywidgets as widgets
from io import StringIO
import sys
import time
from .logging_interface import AgentLogger
import asyncio
from queue import Queue
import threading

class LoggingPanel:
    """A panel for displaying logs in Jupyter notebooks with formatting and styling."""
    
    def __init__(self, title: str = "Log Panel", max_entries: int = 100):
        """Initialize the logging panel.
        
        Args:
            title: Title of the logging panel
            max_entries: Maximum number of log entries to keep
        """
        self.title = title
        self.max_entries = max_entries
        self.entries = []
        self._setup_styles()
        
    def _setup_styles(self):
        """Set up CSS styles for the logging panel."""
        self.styles = """
        <style>
            .log-panel {
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                background-color: #1e1e1e;
                color: #d4d4d4;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                max-height: 500px;
                overflow-y: auto;
            }
            .log-entry {
                margin: 5px 0;
                padding: 5px;
                border-left: 3px solid #4CAF50;
            }
            .log-entry.error {
                border-left-color: #f44336;
                color: #ff6b6b;
            }
            .log-entry.warning {
                border-left-color: #ff9800;
                color: #ffd54f;
            }
            .log-entry.info {
                border-left-color: #2196F3;
                color: #64b5f6;
            }
            .log-timestamp {
                color: #888;
                font-size: 0.8em;
                margin-right: 10px;
            }
            .log-level {
                font-weight: bold;
                margin-right: 10px;
            }
            .log-message {
                white-space: pre-wrap;
                word-break: break-word;
            }
            .log-details {
                margin-left: 20px;
                font-size: 0.9em;
                color: #aaa;
            }
        </style>
        """
        
    def _format_entry(self, entry: Dict[str, Any]) -> str:
        """Format a log entry as HTML.
        
        Args:
            entry: Dictionary containing log entry details
            
        Returns:
            Formatted HTML string
        """
        timestamp = entry.get('timestamp', datetime.now().isoformat())
        level = entry.get('level', 'INFO')
        message = entry.get('message', '')
        details = entry.get('details', {})
        
        # Format the entry
        html = f"""
        <div class="log-entry {level.lower()}">
            <span class="log-timestamp">{timestamp}</span>
            <span class="log-level">{level}</span>
            <span class="log-message">{message}</span>
        """
        
        # Add details if present
        if details:
            html += f"""
            <div class="log-details">
                <pre>{json.dumps(details, indent=2)}</pre>
            </div>
            """
            
        html += "</div>"
        return html
        
    def add_entry(self, level: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Add a new log entry.
        
        Args:
            level: Log level (INFO, WARNING, ERROR)
            message: Log message
            details: Optional dictionary with additional details
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'details': details or {}
        }
        
        self.entries.append(entry)
        
        # Trim entries if exceeding max_entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
            
        self.display()
        
    def info(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Add an INFO level log entry."""
        self.add_entry('INFO', message, details)
        
    def warning(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Add a WARNING level log entry."""
        self.add_entry('WARNING', message, details)
        
    def error(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Add an ERROR level log entry."""
        self.add_entry('ERROR', message, details)
        
    def clear(self):
        """Clear all log entries."""
        self.entries = []
        self.display()
        
    def display(self):
        """Display the current state of the logging panel."""
        # Create the panel HTML
        html = f"""
        {self.styles}
        <div class="log-panel">
            <h3>{self.title}</h3>
            {''.join(self._format_entry(entry) for entry in self.entries)}
        </div>
        """
        
        display(HTML(html))
        
    def get_logger(self) -> logging.Logger:
        """Get a Python logger that writes to this panel.
        
        Returns:
            Configured logging.Logger instance
        """
        logger = logging.getLogger(self.title)
        logger.setLevel(logging.INFO)
        
        class PanelHandler(logging.Handler):
            def __init__(self, panel):
                super().__init__()
                self.panel = panel
                
            def emit(self, record):
                try:
                    msg = self.format(record)
                    level = record.levelname
                    details = {
                        'module': record.module,
                        'function': record.funcName,
                        'line': record.lineno
                    }
                    self.panel.add_entry(level, msg, details)
                except Exception:
                    self.handleError(record)
                    
        handler = PanelHandler(self)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

class BaseLoggingPanel(AgentLogger):
    """Base class for logging panels in Jupyter notebooks"""
    
    def __init__(self, title: str = "Debug Logs"):
        self.title = title
        self.logs = []
        self.output = widgets.Output()
        
        # Create a container with proper layout
        self.container = widgets.VBox([
            widgets.HTML(f"<h3>{title}</h3>"),
            self.output
        ], layout=widgets.Layout(width='100%', height='400px', overflow='auto'))
        
        # Display the container
        display(self.container)
        
        # Log initialization
        self.log_event("INIT", "Logging panel initialized")
    
    def start(self):
        """Start logging session"""
        self.clear()
        self.log_event("START", "Logging session started")
    
    def _format_log(self, level: str, message: str, details: Any = None) -> str:
        """Format a log entry with proper word wrapping"""
        # Format the timestamp and level
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_color = {
            "INFO": "blue",
            "WARNING": "orange",
            "ERROR": "red"
        }.get(level, "black")
        
        # Format the message with word wrap
        formatted_message = f'<div style="white-space: pre-wrap; word-wrap: break-word; margin: 5px 0;">{message}</div>'
        
        # Format the details if present
        formatted_details = ""
        if details:
            if isinstance(details, (dict, list)):
                try:
                    formatted_details = f'<pre style="white-space: pre-wrap; word-wrap: break-word; margin: 5px 0; padding: 5px; background-color: #f5f5f5;">{json.dumps(details, indent=2)}</pre>'
                except:
                    formatted_details = f'<pre style="white-space: pre-wrap; word-wrap: break-word; margin: 5px 0; padding: 5px; background-color: #f5f5f5;">{str(details)}</pre>'
            else:
                formatted_details = f'<pre style="white-space: pre-wrap; word-wrap: break-word; margin: 5px 0; padding: 5px; background-color: #f5f5f5;">{str(details)}</pre>'
        
        # Combine everything
        return f'<div style="margin: 5px 0;"><span style="color: {level_color}; font-weight: bold;">[{timestamp}] {level}:</span>{formatted_message}{formatted_details}</div>'
    
    def _update_display(self):
        """Update the display with all logs"""
        with self.output:
            clear_output(wait=True)
            display(HTML("".join(self.logs)))
    
    def log_event(self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log a structured event"""
        log_entry = self._format_log("INFO", f"\n{event_type}: {message}", data)
        self.logs.append(log_entry)
        self._update_display()
    
    def log_warning(self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log a warning"""
        log_entry = self._format_log("WARNING", f"\n{event_type}: {message}", data)
        self.logs.append(log_entry)
        self._update_display()
    
    def log_error(self, error_type: str, message: str, error: Exception = None):
        """Log an error with details"""
        error_details = {
            "error_type": error_type,
            "error_message": str(error) if error else message
        }
        if error:
            error_details["traceback"] = str(error.__traceback__)
        log_entry = self._format_log("ERROR", f"\n{error_type}: {message}", error_details)
        self.logs.append(log_entry)
        self._update_display()
    
    def log_model_response(self, model_name: str, response: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a model response"""
        details = {"response": response}
        if metadata:
            details.update(metadata)
        log_entry = self._format_log("INFO", f"\n{model_name}:", details)
        self.logs.append(log_entry)
        self._update_display()
    
    def clear(self):
        """Clear all logs"""
        self.logs = []
        self._update_display()
    
    def stop(self):
        """Stop logging"""
        self.clear() 