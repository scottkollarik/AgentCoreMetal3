from typing import List, Dict, Any, Optional
import requests
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime, timedelta

class SearchTools:
    """Tools for searching the web and finding images"""
    
    # Rate limiting configuration
    _last_search_time = datetime.min
    _min_search_interval = 2  # seconds between searches
    
    @classmethod
    def _check_rate_limit(cls) -> None:
        """Check and enforce rate limiting"""
        now = datetime.now()
        time_since_last_search = (now - cls._last_search_time).total_seconds()
        if time_since_last_search < cls._min_search_interval:
            time.sleep(cls._min_search_interval - time_since_last_search)
        cls._last_search_time = datetime.now()
    
    @classmethod
    def set_rate_limit(cls, seconds: int) -> None:
        """Set the minimum interval between searches"""
        cls._min_search_interval = max(1, seconds)  # Minimum 1 second
    
    @staticmethod
    def search_images(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Search for images using DuckDuckGo
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of dictionaries containing image information
        """
        try:
            SearchTools._check_rate_limit()
            with DDGS() as ddgs:
                results = list(ddgs.images(query, max_results=max_results))
                return [
                    {
                        "url": result["image"],
                        "title": result["title"],
                        "source": result["source"]
                    }
                    for result in results
                ]
        except Exception as e:
            print(f"Error searching images: {str(e)}")
            return []
    
    @staticmethod
    def search_recipes(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Search for recipes using DuckDuckGo
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of dictionaries containing recipe information
        """
        try:
            SearchTools._check_rate_limit()
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                recipes = []
                
                for result in results:
                    try:
                        # Try to extract recipe information
                        response = requests.get(result["link"])
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Look for common recipe elements
                        title = soup.find('h1')
                        ingredients = soup.find_all(['li', 'p'], string=lambda x: x and any(word in x.lower() for word in ['ingredients', 'recipe']))
                        instructions = soup.find_all(['li', 'p'], string=lambda x: x and any(word in x.lower() for word in ['instructions', 'directions', 'method']))
                        
                        recipes.append({
                            "title": title.text if title else result["title"],
                            "url": result["link"],
                            "description": result["body"],
                            "ingredients": [i.text for i in ingredients[:5]] if ingredients else [],
                            "instructions": [i.text for i in instructions[:5]] if instructions else []
                        })
                    except Exception as e:
                        print(f"Error parsing recipe: {str(e)}")
                        continue
                
                return recipes
        except Exception as e:
            print(f"Error searching recipes: {str(e)}")
            return []
    
    @staticmethod
    def search_trends(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Search for trend data using DuckDuckGo
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of dictionaries containing trend information
        """
        try:
            SearchTools._check_rate_limit()
            with DDGS() as ddgs:
                results = list(ddgs.text(f"{query} trends data", max_results=max_results))
                trends = []
                
                for result in results:
                    try:
                        # Try to extract trend information
                        response = requests.get(result["link"])
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Look for trend data
                        data = soup.find_all(['table', 'div'], class_=lambda x: x and any(word in str(x).lower() for word in ['trend', 'data', 'chart']))
                        
                        trends.append({
                            "title": result["title"],
                            "url": result["link"],
                            "description": result["body"],
                            "data": [d.text for d in data[:3]] if data else []
                        })
                    except Exception as e:
                        print(f"Error parsing trend data: {str(e)}")
                        continue
                
                return trends
        except Exception as e:
            print(f"Error searching trends: {str(e)}")
            return [] 