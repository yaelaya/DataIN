import traceback
from .llm_service import LlamaTextAnalyzer
from .data_analysis import H2ODataAnalyzer
import pandas as pd
import numpy as np
import logging
from typing import Optional
from .visualization_service import VisualizationService

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        self.h2o_enabled = False
        self.image_generator = None 
        self.visualization_service = VisualizationService()
        
        try:
            self.text_analyzer = LlamaTextAnalyzer()
            logger.info("LLM service initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM service: {str(e)}")
            self.text_analyzer = None
            
        try:
            self.data_analyzer = H2ODataAnalyzer()
            self.h2o_enabled = True
            logger.info("H2O service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize H2O: {str(e)}")
            raise
        

    async def process_dataset(self, df: pd.DataFrame, columns: list, target_column: Optional[str] = None):
        try:
            # Data Analysis
            if self.h2o_enabled: 
                analysis = self.data_analyzer.analyze_dataset(df=df, target_column=target_column)
            else:
                analysis = {
                    "summary": df.describe().to_dict(),
                    "correlation": df.corr().to_dict(),
                    "note": "Using pandas fallback (H2O unavailable)"
                }
            
            insights = None
            if self.text_analyzer:
                sample = df.head(1000)
                data_summary = f"""
                Dataset Overview:
                - Rows: {len(df)}, Columns: {len(columns)}
                - Sample Statistics:
                {sample[columns].describe().to_string()}
                """
                try:
                    insights = await self.text_analyzer.generate_insights(data_summary, columns)
                except Exception as e:
                    logger.error(f"LLM analysis failed: {str(e)}")
                    insights = "LLM analysis unavailable"

            # Generate visualization
            visualization = None
            if len(df.columns) > 1:
                try:
                    numeric_cols = df.select_dtypes(include=np.number).columns
                    if len(numeric_cols) > 0:
                        config = {
                            'plot_type': 'histogram',
                            'column': numeric_cols[0],
                            'title': f'Distribution of {numeric_cols[0]}'
                        }
                        visualization = self.visualization_service.generate_plot(df, 'histogram', config)
                except Exception as e:
                    logger.error(f"Visualization generation failed: {str(e)}")
                    visualization = None

            return {
                "analysis": analysis,
                "insights": insights or "LLM service not available",
                "visualization": visualization,
                "stats": {
                    "rows_processed": len(df),
                    "columns_analyzed": len(columns)
                }
            }
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}\n{traceback.format_exc()}")
            raise

    def _fuzzy_match(self, column_name, available_columns):
        # Simple implementation - adjust as needed
        for col in available_columns:
            if column_name.lower() in col.lower():
                return col
        return None

    async def _generate_response(self, original_question, interpreted, data_result):
        if not self.text_analyzer:
            return "AI service unavailable"
        
        prompt = f"""
        Original question: {original_question}
        Interpreted as: {interpreted}
        Data result: {data_result}
        
        Generate a helpful response to the user's question based on this data.
        """
        return await self.text_analyzer.generate_insights(prompt, [])

    async def answer_question(self, df: pd.DataFrame, raw_question: str):
        # Step 1: NLU Interpretation
        interpretation = await self.understand_query(raw_question, df.columns.tolist())
        
        # Step 2: Column Correction
        corrected_columns = []
        for col in interpretation['target_columns']:
            if col not in df.columns:
                # Find closest match using your data's column names
                corrected = self._fuzzy_match(col, df.columns)
                if corrected: corrected_columns.append(corrected)
        
        # Step 3: Intent Handling
        if interpretation['intent'] == "stats":
            result = self._calculate_stats(df, corrected_columns)
        elif interpretation['intent'] == "filter":
            result = self._filter_data(df, interpretation['time_reference'])
        
        # Step 4: Natural Language Response
        return await self._generate_response(
            original_question=raw_question,
            interpreted=interpretation,
            data_result=result
        )
    

    async def generate_story_insights(self, df: pd.DataFrame, plots: list) -> dict:
        """Generate structured insights for data stories"""
        try:
            # Get basic insights
            insights = await self.generate_insights(
                f"Dataset with {len(df)} rows and {len(df.columns)} columns.\n"
                f"Sample data:\n{df.head(3).to_string()}"
            )
        
            # Structure insights for storytelling
            structured = {
                "insights": insights,
                "highlights": [
                    f"Found {len(plots)} key visualizations",
                    f"Dataset contains {len(df.select_dtypes(include=np.number).columns)} numeric columns",
                    f"Time range: {df['date_column'].min()} to {df['date_column'].max()}" 
                    # Add more dynamic highlights
                ]
            }
            return structured
        
        except Exception as e:
            logger.error(f"Story generation failed: {str(e)}")
            return {"error": str(e)}
        


    async def generate_ai_narrative(self, df: pd.DataFrame, analysis: dict) -> str:
        """Generates a human-like data story"""
        prompt = f"""Transform this analysis into a engaging narrative:
    
        Data Shape: {df.shape}
        Key Stats: {analysis['descriptive_stats']}
        Insights: {analysis['insights']}
    
        Write as a data journalist would, with:
        1. A hook/intro
        2. 3-5 key story beats
        3. Surprising findings
        4. Business implications"""
    
        return await self.text_analyzer.generate_insights(prompt)

    async def generate_annotated_plots(self, df: pd.DataFrame) -> list:
        """Generates plots with narrative captions"""
        plots = []
        for col in df.select_dtypes(include=np.number).columns[:3]:  # First 3 numeric cols
            plot = self.visualization_service.generate_plot(df, 'histogram', {'column': col})
            caption = await self.text_analyzer.generate_insights(
                f"Write a 1-sentence caption about what this histogram of {col} shows:"
            )
            plots.append({
                "plot": plot,
                "caption": caption
            })
        return plots
    

    