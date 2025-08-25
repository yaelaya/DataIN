import os
import h2o
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from routes import analysis

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    try:
        # Proper way to shutdown if any existing instance
        try:
            h2o.shutdown(prompt=False)
        except:
            pass
            
        h2o.init(
            strict_version_check=False,
            max_mem_size="6G",
            nthreads=1,
            port=54321,
            ice_root=os.getenv('TEMP', '/tmp')
        )
        logger.info("H2O initialized successfully")
    except Exception as e:
        logger.error(f"H2O initialization failed: {str(e)}")
        raise RuntimeError(f"H2O initialization failed: {str(e)}")
    
    yield
    
    # Shutdown code
    try:
        h2o.shutdown(prompt=False)
        logger.info("H2O cluster shutdown successfully")
    except Exception as e:
        logger.error(f"Error shutting down H2O: {str(e)}")

app = FastAPI(
    title="AI Data Dashboard",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Timeout middleware
@app.middleware("http")
async def timeout_middleware(request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=500)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

app.include_router(analysis.router, prefix="/api/analysis")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)