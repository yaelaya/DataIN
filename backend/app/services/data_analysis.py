import h2o
import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import os
import traceback  # Add this with other imports

logger = logging.getLogger(__name__)

class H2ODataAnalyzer:
    def __init__(self):
        self.h2o_enabled = False
        self._init_h2o()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _init_h2o(self):
        try:
            # Clean shutdown if any existing instance
            try:
                h2o.shutdown(prompt=False)
            except:
                pass
            
            h2o.init(
                strict_version_check=False,
                max_mem_size="4G",
                nthreads=1,
                port=54321,
                ice_root=os.getenv('TEMP', '/tmp'),
            )
            # Verify connection
            if not h2o.connection().connected:
                raise RuntimeError("H2O cluster not connected")
            
            logger.info("H2O initialized successfully")

        except Exception as e:
            logger.error(f"H2O initialization failed: {str(e)}")
            raise RuntimeError(f"H2O initialization failed: {str(e)}")

    def preprocess_data(self, df):
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
    
        # Convert all NA-containing columns to appropriate types
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('missing')
            else:
                df[col] = df[col].fillna(0).astype('float64')
        return df

    def analyze_dataset(self, df: pd.DataFrame, target_column: str = None, llm_recommendations: dict = None):
        try:
            # Deep copy to avoid modifying original
            df = df.copy(deep=True)
    
            # Convert datetime columns to strings and handle timezone-aware datetimes
            date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns
            for col in date_cols:
                try:
                    df[col] = df[col].astype(str)
                except Exception as e:
                    logger.warning(f"Couldn't convert {col} to string: {str(e)}")
                    df[col] = df[col].apply(lambda x: str(x) if not pd.isna(x) else 'MISSING')
        
            # Handle infinite values and NAs more robustly
            numeric_cols = df.select_dtypes(include=np.number).columns
            non_numeric_cols = df.select_dtypes(exclude=np.number).columns
        
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            df[numeric_cols] = df[numeric_cols].fillna(-999999)
            df[non_numeric_cols] = df[non_numeric_cols].fillna('MISSING')

            # Convert to H2OFrame with enhanced validation
            try:
                
                hf = h2o.H2OFrame(df)
                if hf is None:
                    raise ValueError("H2OFrame conversion returned None")
            
                # Validate dimensions after conversion
                if hf.nrow != df.shape[0]:
                    # Try dropping NA rows if mismatch occurs
                    df_clean = df.dropna()
                    hf = h2o.H2OFrame(df_clean)
                    if hf.nrow != df_clean.shape[0]:
                        raise ValueError(
                            f"Persistent dimension mismatch: Pandas {df_clean.shape} vs H2O {hf.dim}\n"
                            f"Original: {df.shape}"
                        )
                    logger.warning(f"Dimension mismatch resolved by dropping NA rows: {df.shape} -> {df_clean.shape}")
            
                # Validate column names
                if set(hf.columns) != set(df.columns):
                    missing_cols = set(df.columns) - set(hf.columns)
                    extra_cols = set(hf.columns) - set(df.columns)
                    raise ValueError(
                        f"Column mismatch after H2O conversion\n"
                        f"Missing: {missing_cols}\n"
                        f"Extra: {extra_cols}"
                    )

            except Exception as e:
                logger.error(f"H2O frame conversion failed: {str(e)}")
                if not self.h2o_enabled:
                    raise
                logger.warning("Falling back to pandas-only analysis")
                hf = None

            # Get columns to analyze with improved fallback logic
            analysis_cols = []
            try:
                if llm_recommendations and 'recommended_columns' in llm_recommendations:
                    analysis_cols = [col for col in llm_recommendations['recommended_columns'] 
                                    if col in (hf.columns if hf is not None else df.columns)]
            
                if not analysis_cols:  # Fallback if no valid columns
                    analysis_cols = self._get_relevant_numeric_cols(hf if hf is not None else df)
                
                if not analysis_cols:  # Final fallback to all numeric columns
                    analysis_cols = list(numeric_cols)
                
            except Exception as e:
                logger.error(f"Column selection failed: {str(e)}")
                analysis_cols = list(numeric_cols) if len(numeric_cols) > 0 else df.columns.tolist()[:5]

            # Generate results with enhanced error handling
            results = {
                "covariance": None,
                "descriptive_stats": None,
                "summary": None,
                "target_stats": None,
                "data_types": None,
                "llm_metadata": llm_recommendations if llm_recommendations else None,
                "warnings": []
            }

            try:
                analysis_data = hf[analysis_cols] if hf is not None else df[analysis_cols]
                results.update({
                    "covariance": self._safe_covariance(analysis_data),
                    "descriptive_stats": self._get_descriptive_stats(analysis_data),
                    "summary": self._safe_describe(analysis_data),
                    "data_types": self._get_column_types(hf if hf is not None else df),
                })
            
                if target_column and target_column in (hf.columns if hf is not None else df.columns):
                    results["target_stats"] = self._safe_target_stats(
                        hf if hf is not None else df, 
                        target_column
                    )
                
            except Exception as e:
                logger.error(f"Analysis failed for some metrics: {str(e)}")
                results["warnings"].append(f"Partial analysis failure: {str(e)}")
            
                # Try to get at least basic stats
                try:
                    results["summary"] = self._safe_describe(df[analysis_cols])
                except:
                    pass

            return results

        except Exception as e:
            logger.error(f"H2O analysis failed completely: {str(e)}\n{traceback.format_exc()}")
            return self._pandas_fallback(df, e)
    

    def _get_relevant_numeric_cols(self, hf: h2o.H2OFrame) -> list:
        """Fallback method for automatic column selection"""
        numeric_cols = [col for col in hf.columns if hf[col].isnumeric()[0]]
        # Add any additional automatic filtering logic here
        return numeric_cols
    
    def _safe_covariance(self, data):
        try:
            if isinstance(data, h2o.H2OFrame):
                numeric_cols = [col for col in data.columns if data[col].isnumeric()[0]]
                return data[numeric_cols].cov().as_data_frame().to_dict()
            else:  # pandas
                numeric_cols = data.select_dtypes(include=np.number).columns
                return data[numeric_cols].cov().to_dict()
        except Exception as e:
            return {"error": str(e)}  # ✅ Indentation correcte

    def _get_descriptive_stats(self, data):
        try:
            if isinstance(data, h2o.H2OFrame):
                data = data.as_data_frame()
            
            numeric_cols = data.select_dtypes(include=np.number).columns
            numeric_data = data[numeric_cols]
            
            return {
                'mean': numeric_data.mean().to_dict(),
                'median': numeric_data.median().to_dict(),
                'std_dev': numeric_data.std().to_dict(),
                'skewness': numeric_data.skew().to_dict(),
                'kurtosis': numeric_data.kurtosis().to_dict(),
                'min': numeric_data.min().to_dict(),
                'max': numeric_data.max().to_dict()
            }
        except Exception as e:
            logger.warning(f"Could not compute descriptive stats: {str(e)}")
            return {"error": str(e)}
            
    def _safe_describe(self, hf):
        try:
            return hf.describe().as_data_frame(use_multi_thread=True).to_dict()
        except Exception as e:
            return {"error": str(e)}

    def _safe_correlation(self, hf):
        try:
            numeric_cols = [col for col in hf.columns if hf[col].isnumeric()[0]]
            if len(numeric_cols) > 1:
                return hf[numeric_cols].cor().as_data_frame(use_multi_thread=True).to_dict()
            return None
        except Exception as e:
            return {"error": str(e)}

    def _safe_target_stats(self, hf, target_column):
        if target_column and target_column in hf.columns:
            try:
                return hf[target_column].as_data_frame().describe().to_dict()
            except:
                return {"error": f"Could not analyze target column {target_column}"}
        return None

    def _get_column_types(self, hf):
        return {col: hf.type(col) for col in hf.columns}  

    def _pandas_fallback(self, df, error):
        try:
            if isinstance(df, h2o.H2OFrame):  # Check if it's an H2O frame
                df = df.as_data_frame(use_multi_thread=True)  # Convert here
            return {
                "summary": df.describe().to_dict(),
                "correlation": df.select_dtypes(include=np.number).corr().to_dict(),
                "error": str(error)
            }
        except Exception as e:
            return {"critical_error": str(e)}
        
    

    # In your H2ODataAnalyzer class
    def _is_numeric(self, column_data):
        """Check if a column contains numeric data (works for both H2OFrame and pandas)"""
        if hasattr(column_data, 'isnumeric'):  # For H2OFrame
            return column_data.isnumeric()[0,0]
        else:  # For pandas
            return pd.api.types.is_numeric_dtype(column_data)
    
