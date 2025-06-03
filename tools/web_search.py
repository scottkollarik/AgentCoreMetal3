from typing import List, Optional
from langchain.tools import Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

class WebSearchTool:
    """Tool for performing web searches"""
    
    def __init__(self, max_results: int = 5):
        self.search = DuckDuckGoSearchAPIWrapper()
        self.max_results = max_results
    
    def create_tool(self) -> Tool:
        """Create a LangChain tool for web search"""
        return Tool(
            name="web_search",
            description="Search the web for information about a topic",
            func=self._search
        )
    
    def _search(self, query: str) -> str:
        """Perform a web search and return formatted results"""
        try:
            results = self.search.results(query, self.max_results)
            formatted_results = []
            
            for result in results:
                formatted_results.append(
                    f"Title: {result['title']}\n"
                    f"Link: {result['link']}\n"
                    f"Snippet: {result['snippet']}\n"
                )
            
            return "\n\n".join(formatted_results)
        except Exception as e:
            return f"Error performing web search: {str(e)}" 