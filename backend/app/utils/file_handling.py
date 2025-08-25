import pandas as pd
from io import BytesIO
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def read_uploaded_file(file_content: bytes, filename: str) -> Tuple[pd.DataFrame, str]:
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(file_content))
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(BytesIO(file_content))
        else:
            raise ValueError("Unsupported file type")
            
        return df, "File read successfully"
    except Exception as e:
        logger.error(f"File reading failed: {str(e)}")
        raise ValueError(f"Failed to read file: {str(e)}")