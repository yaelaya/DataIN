import base64
import io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import logging
import json

logger = logging.getLogger(__name__)

class VisualizationService:
    def __init__(self):
        self.available_plots = {
            'histogram': self.generate_histogram,
            'scatter': self.generate_scatter,
            'box': self.generate_boxplot,
            'heatmap': self.generate_heatmap,
            'line': self.generate_line,
        }
    
    def generate_plot(self, df: pd.DataFrame, plot_type: str, config: Dict) -> str:
        """Generate plot based on type and configuration"""
        try:
            if plot_type not in self.available_plots:
                raise ValueError(f"Unsupported plot type: {plot_type}")
        
            plot_func = self.available_plots[plot_type]
            fig = plot_func(df, config)
        
        # Try to export as PNG first, fall back to HTML if Chrome isn't available
            try:
                img_bytes = fig.to_image(format="png")
                return base64.b64encode(img_bytes).decode('utf-8')
            except Exception as e:
                logger.warning(f"PNG export failed, falling back to HTML: {str(e)}")
                html_str = fig.to_html(full_html=False, include_plotlyjs='cdn')
                return base64.b64encode(html_str.encode()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Failed to generate {plot_type} plot: {str(e)}")
            raise ValueError(f"Failed to generate {plot_type} plot: {str(e)}")




    def generate_histogram(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Generate histogram with customization"""
        if 'column' not in config or config['column'] not in df.columns:
            raise ValueError("Histogram requires a valid 'column' parameter")
    
        # Use default 20 bins if not specified or if 'auto' was provided
        bins = config.get('bins', 20)
        if bins == 'auto':
            bins = 20  # Default number of bins
        
        fig = px.histogram(
            df,
            x=config['column'],
            nbins=int(bins),  # Ensure it's an integer
            color=config.get('color_column'),
            title=config.get('title', f"Histogram of {config['column']}"),
            marginal=config.get('marginal')
        )
        self._apply_layout_config(fig, config)
        return fig




    def generate_scatter(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Generate scatter plot with customization"""
        fig = px.scatter(
            df,
            x=config['x_column'],
            y=config['y_column'],
            color=config.get('color_column'),
            size=config.get('size_column'),
            title=config.get('title', f"{config['y_column']} vs {config['x_column']}"),
            labels={
                config['x_column']: config.get('xlabel', config['x_column']),
                config['y_column']: config.get('ylabel', config['y_column'])
            },
            trendline=config.get('trendline')
        )
        self._apply_layout_config(fig, config)
        return fig



    def generate_boxplot(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Generate box plot with customization"""
        if 'y_column' not in config or config['y_column'] not in df.columns:
            raise ValueError("Boxplot requires a valid 'y_column' parameter")
    
        # Use x_column if provided, otherwise make a single box
        if 'x_column' in config and config['x_column']:
            fig = px.box(
                df,
                x=config['x_column'],
                y=config['y_column'],
                color=config.get('color_column'),
                title=config.get('title', f"Boxplot of {config['y_column']}")
            )
        else:
            fig = px.box(
                df,
                y=config['y_column'],
                title=config.get('title', f"Boxplot of {config['y_column']}")
            )
    
        self._apply_layout_config(fig, config)
        return fig




    def generate_heatmap(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Generate correlation heatmap"""
        numeric_df = df.select_dtypes(include=np.number)
        if len(numeric_df.columns) < 2:
            raise ValueError("Need at least 2 numeric columns for heatmap")
            
        corr_matrix = numeric_df.corr()
        
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale=config.get('colorscale', 'Viridis')
            )
        )
        
        fig.update_layout(
            title=config.get('title', "Correlation Heatmap")
        )
        self._apply_layout_config(fig, config)
        return fig




    def generate_line(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Generate line plot"""
        if 'x_column' not in config or 'y_column' not in config:
            raise ValueError("Line plot requires both x_column and y_column")
            
        fig = px.line(
            df,
            x=config['x_column'],
            y=config['y_column'],
            color=config.get('color_column'),
            title=config.get('title', f"{config['y_column']} over {config['x_column']}")
        )
        self._apply_layout_config(fig, config)
        return fig




    def _apply_layout_config(self, fig: go.Figure, config: Dict) -> None:
        """Apply layout customization"""
        layout_updates = {}
        
        if 'width' in config:
            layout_updates['width'] = config['width']
        if 'height' in config:
            layout_updates['height'] = config['height']
        if 'font_size' in config:
            layout_updates['font'] = {'size': config['font_size']}
            
        fig.update_layout(**layout_updates)


