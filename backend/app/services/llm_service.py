import json
import pandas as pd
import requests
import logging
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class LlamaTextAnalyzer:
    def __init__(self):
        self.base_url = "http://localhost:11434/api"
        logger.info("LLM Analyzer initialized")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_insights(self, data_summary: str, columns: List[str]) -> str:
        try:
            prompt = f"""Analyze this dataset:
            {data_summary}
            Focus on columns: {', '.join(columns)}
            Provide concise bullet points of key findings."""
            
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "model": "tinyllama",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=500
            )
            response.raise_for_status()
            return response.json().get("response", "No insights generated")
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            return f"Insights generation failed: {str(e)}"
        



    # llm_service.py
async def answer_from_full_data(self, question: str, full_analysis: dict, full_data: pd.DataFrame):
    # Step 1: Create comprehensive data profile
    data_profile = {
        "shape": full_data.shape,
        "dtypes": str(full_data.dtypes.to_dict()),
        "stats": full_analysis['analysis']['summary'],
        "sample": full_data.sample(min(100, len(full_data))).to_dict('records') 
    }
    
    # Step 2: Dynamic context window optimization
    if len(full_data) > 10_000:
        context_strategy = "aggregates"
        context_data = {
            "top_values": {col: full_data[col].value_counts().head(3).to_dict() 
                         for col in full_data.columns}
        }
    else:
        context_strategy = "full_sample"
        context_data = {"records": full_data.to_dict('records')}

    # Step 3: Structured prompt
    prompt = f"""Analyze this COMPLETE dataset:
    - Profile: {json.dumps(data_profile)}
    - Context Strategy: {context_strategy}
    - Data: {json.dumps(context_data)}
    
    Question: {question}
    
    Answer precisely using:""" + "\n1. Exact stats when available\n2. Full data patterns\n3. Avoid approximations"
    
    # Step 4: LLM call with your existing retry logic
    return self._call_llm(prompt)

async def understand_query(self, raw_question: str, df_columns: list) -> dict:
    """
    Interprets messy user input and connects it to the dataset.
    Returns structured understanding.
    """
    prompt = f"""Interpret this data question:
    
    User Input: "{raw_question}"
    Available Columns: {', '.join(df_columns)}
    
    Output JSON with:
    - "intent": ["stats", "filter", "calculation"]
    - "target_columns": []
    - "possible_misspellings": {"user_word": "column_match"}
    - "time_reference": ("range"/"point"/None)
    """
    
    response = await self._call_llm(prompt)
    return self._validate_interpretation(json.loads(response), df_columns)