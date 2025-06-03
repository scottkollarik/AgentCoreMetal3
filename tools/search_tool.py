from langchain.tools import BaseTool
from langchain.tools import DuckDuckGoSearchRun
from typing import Optional, Type, Any

class SearchTool(BaseTool):
    """Tool for searching the internet using DuckDuckGo"""
    
    def __init__(self):
        search = DuckDuckGoSearchRun()
        super().__init__(
            name="search",
            description="Useful for searching the internet for information about a specific topic",
            func=search.run,
            coroutine=None
        )
    
    def _run(self, query: str) -> str:
        """Run the search tool synchronously"""
        search = DuckDuckGoSearchRun()
        return search.run(query)
    
    async def _arun(self, query: str) -> str:
        """Run the search tool asynchronously"""
        return self._run(query) 