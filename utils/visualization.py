from typing import List, Dict, Any, Optional, Union
import graphviz
import plotly.graph_objects as go
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from IPython.display import display, Image as IPImage
import plotly.express as px
import json

class VisualizationTools:
    """Tools for displaying images and generating charts"""
    
    @staticmethod
    def display_image(url: str, width: int = 400) -> None:
        """Display an image from a URL
        
        Args:
            url: URL of the image to display
            width: Width of the displayed image in pixels
        """
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            display(IPImage(data=response.content, width=width))
        except Exception as e:
            print(f"Error displaying image: {str(e)}")
    
    @staticmethod
    def display_images(urls: List[str], width: int = 400) -> None:
        """Display multiple images from URLs
        
        Args:
            urls: List of image URLs to display
            width: Width of each displayed image in pixels
        """
        for url in urls:
            VisualizationTools.display_image(url, width)
    
    @staticmethod
    def create_line_chart(data: List[Dict[str, Any]], 
                         x_field: str = "year",
                         y_field: str = "popularity",
                         color_field: Optional[str] = None,
                         title: str = "Trend Over Time") -> None:
        """Create a line chart using Plotly
        
        Args:
            data: List of dictionaries containing the data
            x_field: Field name for x-axis values
            y_field: Field name for y-axis values
            color_field: Optional field name to use for line colors
            title: Chart title
        """
        df = pd.DataFrame(data)
        
        fig = px.line(df, 
                     x=x_field, 
                     y=y_field,
                     color=color_field if color_field else None,
                     title=title)
        
        fig.update_layout(
            xaxis_title=x_field.capitalize(),
            yaxis_title=y_field.capitalize(),
            showlegend=True
        )
        
        fig.show()
    
    @staticmethod
    def create_bar_chart(data: List[Dict[str, Any]],
                        x_field: str,
                        y_field: str,
                        title: str = "Comparison Chart") -> None:
        """Create a bar chart using Plotly
        
        Args:
            data: List of dictionaries containing the data
            x_field: Field name for x-axis values
            y_field: Field name for y-axis values
            title: Chart title
        """
        df = pd.DataFrame(data)
        
        fig = px.bar(df,
                    x=x_field,
                    y=y_field,
                    title=title)
        
        fig.update_layout(
            xaxis_title=x_field.capitalize(),
            yaxis_title=y_field.capitalize()
        )
        
        fig.show()
    
    @staticmethod
    def create_trend_chart(data: Union[str, List[Dict[str, Any]]],
                          title: str = "Trend Analysis") -> None:
        """Create a trend chart from JSON data
        
        Args:
            data: JSON string or list of dictionaries containing trend data
            title: Chart title
        """
        # Convert string to list if needed
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                print("Error: Invalid JSON data")
                return
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create the chart
        fig = go.Figure()
        
        # Add a line for each shape
        for shape in df['Shape'].unique():
            shape_data = df[df['Shape'] == shape]
            fig.add_trace(go.Scatter(
                x=shape_data['Year'],
                y=shape_data['QuantitySold'],
                name=shape,
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title="Quantity Sold",
            showlegend=True
        )
        
        fig.show() 