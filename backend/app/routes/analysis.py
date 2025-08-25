from fastapi import APIRouter, Form, UploadFile, File, HTTPException
from services.ai_service import AIService
import pandas as pd
import numpy as np
import logging
import traceback
from typing import Any
import math 
import json
import httpx
from fastapi.responses import JSONResponse 
import requests
from services.visualization_service import VisualizationService
from typing import Dict
import matplotlib.pyplot as plt 
import base64
import seaborn as sns
import io
from datetime import datetime
import asyncio
from fastapi import BackgroundTasks
import uuid  # For generating job IDs
from typing import Dict  # For type hints


logger = logging.getLogger(__name__)
router = APIRouter()
ai_service = AIService()
vis_service = VisualizationService()


job_status: Dict[str, Dict] = {}


def clean_for_json(data: Any) -> Any:
    """
    Recursively clean data for JSON serialization while preserving numeric types
    """
    if isinstance(data, dict):
        return {k: clean_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_for_json(v) for v in data]
    elif isinstance(data, (float, np.floating)):
        if math.isnan(data) or math.isinf(data):
            return None
        return float(data)
    elif isinstance(data, (np.integer, int)):
        return int(data)
    elif isinstance(data, pd.DataFrame):
        return clean_for_json(data.to_dict(orient="records"))
    return data


@router.post("/analyze")
async def analyze_data(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not (file.filename.endswith('.csv') or file.filename.endswith(('.xlsx', '.xls'))):
            raise HTTPException(status_code=400, detail="Supported formats: CSV, Excel (.xlsx, .xls)")

        # Read file with error handling
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file.file)
            else:
                df = pd.read_excel(file.file, engine='openpyxl')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"File reading failed: {str(e)}")

        # Basic data validation
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(df.columns) < 2:
            raise HTTPException(status_code=400, detail="Dataset must have at least 2 columns")

        # Clean data before processing
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna('MISSING')  # For string columns
        df = df.fillna(-999)       # For numeric columns


        numeric_cols = df.select_dtypes(include=np.number).columns
        non_numeric_cols = df.select_dtypes(exclude=np.number).columns

        df[numeric_cols] = df[numeric_cols].fillna(0)
        df[non_numeric_cols] = df[non_numeric_cols].fillna('missing')

        # Process dataset with enhanced error handling
        try:
            result = await ai_service.process_dataset(
                df=df,
                columns=df.columns.tolist()
            )
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail="Data analysis failed")


        sample_data = df.head(10).copy()
        sample_data = sample_data.replace([np.nan], None)  # Only replace remaining NaNs

        # Prepare the response with sample data
        response = {
            "analysis": {
                "covariance": df[numeric_cols].cov().to_dict(),
                "descriptive_stats": {
                    'mean': df[numeric_cols].mean().to_dict(),
                    'median': df[numeric_cols].median().to_dict(),
                    'std': df[numeric_cols].std().to_dict(),
                    'min': df[numeric_cols].min().to_dict(),
                    'max': df[numeric_cols].max().to_dict()
                },
                "summary": df.describe().to_dict()
            },
            "insights": result.get("insights", "No insights generated"),
            "row_count": len(df),
            "columns": df.columns.tolist(),
            "numeric_columns": numeric_cols.tolist(),
            "sample": clean_for_json(sample_data.to_dict(orient="records"))
        }

        # Add visualization if available
        if "visualization" in result:
            response["visualization"] = result["visualization"]
        else:
            response["visualization_status"] = "Visualization generation failed"

        # Clean the final response before returning
        clean_response = clean_for_json(response)
        return JSONResponse(content=clean_response)

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")     



@router.post("/ask-question")
async def ask_question(file: UploadFile = File(...), question: str = Form(...)):
    try:
        logger.info(f"Received question: '{question}' for file: {file.filename}")
        
        # Read file
        try:
            if file.filename.lower().endswith('.csv'):
                df = pd.read_csv(file.file)
            else:
                df = pd.read_excel(file.file, engine='openpyxl')
        except pd.errors.EmptyDataError:
            return {"answer": "Error: Uploaded file is empty"}
        
        # Prepare data summary
        data_summary = {
            "rows": len(df),
            "columns": df.columns.tolist(),
            "sample": df.head(2).to_dict(orient='records')
        }
        
        # Call LLM
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "tinyllama",
                    "prompt": f"Data: {data_summary}\nQuestion: {question}\nAnswer:",
                    "stream": False
                },
                timeout=500.0
            )
            response.raise_for_status()
            result = response.json()
            
            return {"answer": result.get("response", "No response generated")}
            
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}", exc_info=True)
        return {"answer": f"Error processing your question: {str(e)}"}




@router.post("/generate-plot")
async def generate_plot(
    file: UploadFile = File(...),
    plot_type: str = Form(...),
    config: str = Form("{}")
):
    try:
        # Read the file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        else:
            df = pd.read_excel(file.file, engine='openpyxl')
            
        plot_config = json.loads(config)
        
        # Validate plot type
        valid_plot_types = ['scatter', 'histogram', 'box', 'line', 'bar', 'pie']
        if plot_type not in valid_plot_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid plot type. Must be one of: {', '.join(valid_plot_types)}"
            )
            
        # Convert date columns
        for col in df.select_dtypes(include=['datetime']).columns:
            df[col] = df[col].dt.strftime('%Y-%m-%d')
            
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        if plot_type == "histogram":
            if 'x_column' not in plot_config:
                raise HTTPException(status_code=400, detail="x_column is required for histogram")
                
            if 'y_column' in plot_config and plot_config['y_column']:
                # 2D histogram
                sns.histplot(
                    data=df,
                    x=plot_config['x_column'],
                    y=plot_config['y_column'],
                    bins=plot_config.get('bins', 30),
                    kde=plot_config.get('density', False),
                    stat="density" if plot_config.get('density', False) else "count",
                    cumulative=plot_config.get('cumulative', False),
                    cbar=True,
                    ax=ax
                )
            else:
                # 1D histogram
                sns.histplot(
                    data=df,
                    x=plot_config['x_column'],
                    bins=plot_config.get('bins', 30),
                    kde=plot_config.get('density', False),
                    stat="density" if plot_config.get('density', False) else "count",
                    cumulative=plot_config.get('cumulative', False),
                    ax=ax
                )
                
        elif plot_type == "scatter":
            if 'x_column' not in plot_config or 'y_column' not in plot_config:
                raise HTTPException(status_code=400, detail="x_column and y_column are required for scatter plot")
                
            sns.scatterplot(
                data=df,
                x=plot_config['x_column'],
                y=plot_config['y_column'],
                hue=plot_config.get('color_column'),
                size=plot_config.get('size_column'),
                ax=ax
            )
            
        elif plot_type == "box":
            if 'y_column' not in plot_config:
                raise HTTPException(status_code=400, detail="y_column is required for box plot")
                
            if 'x_column' in plot_config and plot_config['x_column']:
                sns.boxplot(
                    data=df,
                    x=plot_config['x_column'],
                    y=plot_config['y_column'],
                    ax=ax
                )
            else:
                sns.boxplot(
                    data=df,
                    y=plot_config['y_column'],
                    ax=ax
                )
                
        elif plot_type == "line":
            if 'x_column' not in plot_config or 'y_column' not in plot_config:
                raise HTTPException(status_code=400, detail="x_column and y_column are required for line plot")
                
            sns.lineplot(
                data=df,
                x=plot_config['x_column'],
                y=plot_config['y_column'],
                ax=ax
            )
            
        elif plot_type == "bar":
            if 'x_column' not in plot_config or 'y_column' not in plot_config:
                raise HTTPException(status_code=400, detail="x_column and y_column are required for bar chart")
                
            sns.barplot(
                data=df,
                x=plot_config['x_column'],
                y=plot_config['y_column'],
                ax=ax
            )
            
        elif plot_type == "pie":
            if 'values_column' not in plot_config or 'labels_column' not in plot_config:
                raise HTTPException(status_code=400, detail="values_column and labels_column are required for pie chart")
                
            pie_data = df.groupby(plot_config['labels_column'])[plot_config['values_column']].sum()
            plt.pie(
                pie_data.values,
                labels=pie_data.index,
                autopct='%1.1f%%',
                startangle=90
            )
            plt.axis('equal')

        # Set title and formatting
        plt.title(plot_config.get('title', f"{plot_type} plot"))
        if len(ax.get_xticklabels()) > 5:
            plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return {
            "plot": base64.b64encode(buf.read()).decode('utf-8'),
            "status": "success",
            "config": plot_config
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plot generation failed: {str(e)}\n{traceback.format_exc()}")
        error_detail = {
            "error": str(e),
            "plot_type": plot_type,
            "config": plot_config,
            "available_columns": list(df.columns) if 'df' in locals() else []
        }
        raise HTTPException(status_code=400, detail=error_detail)
        
         

@router.post("/generate-story")
async def generate_data_story(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Read file content immediately (before passing to background task)
        file_content = await file.read()

        # Initialize job status with more detailed structure
        job_status[job_id] = {
            "status": "processing",
            "progress": 0,
            "stage": "initializing",
            "result": None,
            "error": None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Start background task with error handling
        try:
            background_tasks.add_task(
                process_story_generation,
                file_content=file_content,
                filename=file.filename,
                job_id=job_id
            )
        except Exception as e:
            logger.error(f"Failed to start background task: {str(e)}")
            job_status[job_id].update({
                "status": "failed",
                "error": f"Background task initialization failed: {str(e)}"
            })
            raise HTTPException(500, detail="Failed to start story generation")
        
        return JSONResponse({
            "job_id": job_id,
            "status": "processing",
            "message": "Story generation started successfully",
            "check_status_at": f"/api/analysis/story-status/{job_id}"
        })
        
    except Exception as e:
        logger.error(f"Story initialization failed: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=f"Initialization error: {str(e)}")
    


# Add this status endpoint
@router.get("/story-status/{job_id}")
async def get_story_status(job_id: str):
    if job_id not in job_status:
        raise HTTPException(404, detail="Job not found")
    
    status = job_status[job_id]
    
    # Clean up old failed jobs (older than 1 hour)
    if status["status"] in ["failed", "complete"]:
        job_time = datetime.fromisoformat(status["timestamp"])
        if (datetime.utcnow() - job_time).total_seconds() > 3600:
            del job_status[job_id]
            raise HTTPException(404, detail="Job expired")
    
    return JSONResponse(status)



async def process_story_generation(file_content: bytes, filename: str, job_id: str):
    """Background task for heavy processing"""
    try:
        # Update job status
        job_status[job_id].update({
            "stage": "reading_file",
            "progress": 5
        })

        # Convert bytes back into a file-like object for pandas
        file_like = io.BytesIO(file_content)
        
        # Read file based on extension
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file_like)
            else:
                df = pd.read_excel(file_like, engine='openpyxl')
        except Exception as e:
            logger.error(f"Failed to read file: {str(e)}")
            raise ValueError(f"File reading failed: {str(e)}")

        job_status[job_id].update({
            "stage": "data_cleaning",
            "progress": 10
        })

        # ---- DATA CLEANING ----
        # [Keep your existing data cleaning code]
        
        # ---- ANALYSIS ----
        job_status[job_id].update({
            "stage": "running_analysis",
            "progress": 30
        })

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        analysis = {
            "numeric_columns": numeric_cols,
            "analysis": {
                "descriptive_stats": {
                    "mean": df[numeric_cols].mean().to_dict(),
                    "median": df[numeric_cols].median().to_dict(),
                    "std": df[numeric_cols].std().to_dict(),
                }
            }
        }

        # Generate recommendations
        try:
            job_status[job_id].update({
                "stage": "generating_recommendations",
                "progress": 50
            })
            recommendations = await generate_recommendations(df, analysis)
            analysis["insights"] = recommendations or "No specific insights generated"
        except Exception as e:
            logger.warning(f"Recommendation generation failed: {str(e)}")
            analysis["insights"] = [
                "Basic analysis completed",
                "Detailed recommendations could not be generated"
            ]

        # Generate slides
        try:
            job_status[job_id].update({
                "stage": "generating_slides",
                "progress": 70
            })
            slides = await generate_presentation_slides(df, analysis, filename)
        except Exception as e:
            logger.error(f"Slide generation failed: {str(e)}")
            slides = [{
                "title": "Analysis Report",
                "type": "title",
                "content": ["Basic analysis completed"]
            }]

        # Prepare final result
        result = {
            "presentation": {
                "slides": slides,
                "theme": "corporate_blue",
                "transition": "slide"
            },
            **analysis
        }

        job_status[job_id].update({
            "status": "complete",
            "progress": 100,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Background generation failed: {str(e)}", exc_info=True)
        job_status[job_id].update({
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })



async def generate_recommendations(df: pd.DataFrame, analysis: dict) -> list:
    """Generate actionable recommendations with proper fallbacks"""
    try:
        # Get stats with fallbacks
        stats = analysis.get('analysis', {}).get('descriptive_stats', {})
        
        # Prepare basic data summary
        data_summary = {
            "shape": f"{len(df)} rows, {len(df.columns)} columns",
            "numeric_columns": analysis.get('numeric_columns', []),
            "sample_stats": {
                "means": stats.get('mean', {}),
                "medians": stats.get('median', {}),
                "std_devs": stats.get('std', {})
            },
            "sample_data": df.head(2).to_dict(orient='records')
        }

        # First try to get recommendations from insights if available
        insights = analysis.get('insights', '')
        if insights and isinstance(insights, str) and len(insights.strip()) > 0:
            try:
                # Parse insights to extract recommendations
                recommendations = []
                insight_lines = insights.split('\n')
                
                # Look for actionable items in insights
                for line in insight_lines:
                    line = line.strip()
                    if line.startswith(('-', '*', '•')) or 'recommend' in line.lower():
                        # Clean up the recommendation text
                        clean_line = line.lstrip('-*• ').strip()
                        if clean_line:
                            recommendations.append(clean_line)
                
                if recommendations:
                    return recommendations[:5]  # Return max 5 recommendations
            except Exception as e:
                logger.warning(f"Failed to parse insights for recommendations: {str(e)}")

        # If no recommendations from insights, generate default ones based on data characteristics
        recommendations = []
        
        # Add recommendations based on data size
        if len(df) > 10000:
            recommendations.append("Consider sampling the data for faster exploratory analysis")
        elif len(df) < 100:
            recommendations.append("The dataset is quite small - consider collecting more data for robust analysis")
        
        # Add recommendations based on numeric columns
        numeric_cols = analysis.get('numeric_columns', [])
        if len(numeric_cols) > 5:
            recommendations.append("Multiple numeric columns available - examine correlations between variables")
        elif len(numeric_cols) == 0:
            recommendations.append("No numeric columns found - consider feature engineering to create numeric features")
        
        # Add recommendations based on missing values
        if df.isnull().sum().sum() > 0:
            recommendations.append("Dataset contains missing values - consider imputation or removal")
        
        # Add general recommendations if we don't have enough
        if len(recommendations) < 3:
            recommendations.extend([
                "Create visualizations to better understand variable distributions",
                "Check for outliers that might affect your analysis",
                "Consider feature selection to identify the most important variables"
            ])
        
        return recommendations[:5]  # Return max 5 recommendations

    except Exception as e:
        logger.error(f"Recommendation generation failed: {str(e)}", exc_info=True)
        return [
            "Could not generate specific recommendations",
            "Please examine the data statistics for insights",
            "Consider visualizing key variables"
        ]
    

    
async def generate_presentation_slides(df: pd.DataFrame, analysis: dict, filename: str) -> list:
    """Generate presentation slides from analysis results"""
    slides = []
    
    # Ensure we have numeric columns
    numeric_cols = analysis.get('numeric_columns', [])
    if not numeric_cols:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Get stats with fallbacks
    stats = analysis.get('analysis', {}).get('descriptive_stats', {})
    mean_values = stats.get('mean', {})
    median_values = stats.get('median', {})
    std_values = stats.get('std', {})

    # Title Slide
    slides.append({
        "title": "Data Analysis Report",
        "subtitle": f"Analysis of {filename}",
        "type": "title",
        "content": [
            f"Rows: {len(df)}, Columns: {len(df.columns)}",
            f"Numeric Columns: {len(numeric_cols)}"
        ]
    })
    
    # Slide 2: Key Stats
    slides.append({
        "title": "Key Statistics",
        "type": "table",
        "content": {
            "headers": ["Statistic"] + numeric_cols[:5],  # Show first 5 cols
            "rows": [
                ["Mean"] + [mean_values.get(col, 'N/A') for col in numeric_cols[:5]],
                ["Median"] + [median_values.get(col, 'N/A') for col in numeric_cols[:5]],
                ["Std Dev"] + [std_values.get(col, 'N/A') for col in numeric_cols[:5]]
            ]
        }
    })
    
    # Slide 3: Data Overview
    slides.append({
        "title": "Data Overview",
        "type": "table",
        "content": {
            "headers": ["Metric", "Value"],
            "rows": [
                ["Total Records", len(df)],
                ["Columns", len(df.columns)],
                ["Numeric Columns", len(numeric_cols)],
                ["Date Columns", len(df.select_dtypes(include=['datetime']).columns)]
            ]
        }
    })
    
    # Slide 4-6: Visualizations
    for i, col in enumerate(numeric_cols[:3]):  # First 3 numeric columns
        try:
            # Determine the best plot type based on data characteristics
            unique_values = df[col].nunique()
            
            if unique_values > 10:
                # For high cardinality - use histogram
                plt.figure(figsize=(10, 6))
                if pd.api.types.is_numeric_dtype(df[col]):
                    sns.histplot(data=df, x=col, bins=30)
                    ylabel = "Count"
                else:
                    sns.countplot(data=df, x=col)
                    ylabel = "Count"
                    plt.xticks(rotation=45)
                plt.ylabel(ylabel)
                plot_type = "histogram"
            else:
                # For low cardinality - use boxplot
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df, y=col)
                plot_type = "boxplot"
            
            plt.title(f'Distribution of {col}')
            plt.tight_layout()
            
            # Save plot to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            
            slides.append({
                "title": f"Distribution: {col}",
                "type": "visualization",
                "content": {
                    "plot": base64.b64encode(buf.read()).decode('utf-8'),
                    "caption": f"{plot_type} showing distribution of {col} values"
                }
            })
        except Exception as e:
            logger.error(f"Failed to generate plot for {col}: {str(e)}")
            continue
    
    # Slide 7: Recommendations
    try:
        slides.append({
            "title": "Recommendations",
            "type": "bullets",
            "content": await generate_recommendations(df, analysis)
        })
    except Exception as e:
        logger.error(f"Failed to generate recommendations: {str(e)}")
        slides.append({
            "title": "Recommendations",
            "type": "bullets",
            "content": ["Could not generate recommendations"]
        })
    
    return slides



async def generate_key_findings(df: pd.DataFrame, analysis: dict) -> list:
    """Generate 3-5 key findings as bullet points"""
    prompt = f"""Based on this data analysis, extract 3-5 key findings as concise bullet points:
    
    Data Shape: {df.shape}
    Key Stats: {analysis['descriptive_stats']}
    Insights: {analysis['insights']}
    
    Format as:
    - [Finding 1]
    - [Finding 2]
    - [Finding 3]"""
    
    response = await text_analyzer.generate_insights(prompt)
    return [line.strip('- ').strip() for line in response.split('\n') if line.strip()]




# Add to your FastAPI router
@router.get("/debug-story")
async def debug_story():
    test_data = {
        "narrative": "TEST: Our data shows 3 key trends...\n1. Revenue peaks in Q3\n2. Coastal regions outperform\n3. New users grow 20% MoM",
        "annotated_plots": [{
            "plot": "<BASE64_TEST_IMAGE>",  # Replace with real base64 or None
            "caption": "TEST: This histogram shows normal distribution"
        }],
        "key_takeaways": [
            "TEST: Prioritize Q3 marketing",
            "TEST: Expand coastal inventory"
        ]
    }
    return JSONResponse(content=test_data) 




@router.post("/generate-solutions")
async def generate_solutions(insights: str = Form(...)):
    prompt = f"""Convert these insights into actionable solutions:
    Insights: {insights}
    
    Respond in this format:
    {{
      "problem": "Summarized issue",
      "solution": "Technical steps to resolve it",
      "tools": "Python libraries or methods",
      "urgency": "Priority level"
    }}"""
    
    response = await tinyllama.generate(prompt)
    return {"solutions": response}


@router.post("/generate-actionable-recommendations")
async def generate_action_recommendations(
    file: UploadFile = File(None),
    insights: str = Form(...),
    numeric_columns: str = Form(""),
    row_count: int = Form(0)
):
    """
    Generates technical, actionable recommendations based on insights
    with direct implementation steps.
    """
    try:
        # Prepare dataset context if file exists
        dataset_context = ""
        if file:
            df = pd.read_csv(file.file) if file.filename.endswith('.csv') else pd.read_excel(file.file)
            dataset_context = f"""
            Dataset Details:
            - Shape: {len(df)} rows, {len(df.columns)} cols
            - Numeric: {df.select_dtypes(include=np.number).columns.tolist()}
            - Sample Stats:
              {df.describe().to_dict()}
            """

        prompt = f"""**Role**: You're a Principal Data Scientist reviewing analysis findings.

**Insights**:
{insights}

**Context**:
{dataset_context}
- Numeric Columns: {numeric_columns}
- Total Rows: {row_count}

**Task**: Generate 3-5 technical recommendations with:
1. Action: Specific technical implementation
2. Implementation: Exact steps/code snippets
3. Rationale: How it addresses the insight
4. Priority: Critical/High/Medium/Low
5. Type: visualization|cleaning|feature_eng|modeling|validation

**Format Example**:
```json
{{
  "recommendations": [
    {{
      "action": "Perform outlier detection on column X",
      "implementation": "Use sklearn's IsolationForest or IQR method",
      "rationale": "Insight suggests anomalous values distorting analysis",
      "priority": "High",
      "type": "cleaning"
    }}
  ]
}}
```"""

        # Call to your LLM (TinyLLAMA in this case)
        llm_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "tinyllama",
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.3}  # More deterministic
            }
        ).json()

        # Parse and validate
        try:
            result = json.loads(llm_response.get("response", "{}"))
            if not result.get("recommendations"):
                raise ValueError("No recommendations generated")
                
            return {
                "status": "success",
                "recommendations": result["recommendations"]
            }
            
        except Exception as e:
            # Fallback to simple generation if parsing fails
            return {
                "status": "partial",
                "recommendations": [{
                    "action": "Investigate key variables",
                    "implementation": f"Analyze distributions of: {numeric_columns.split(',')[:3]}",
                    "rationale": "Basic exploratory data analysis",
                    "priority": "Medium",
                    "type": "visualization"
                }]
            }

    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "fallback_recommendations": [{
                "action": "Review data quality",
                "implementation": "Check for missing values and outliers",
                "rationale": "Ensures reliable analysis",
                "priority": "High",
                "type": "cleaning"
            }]
        }