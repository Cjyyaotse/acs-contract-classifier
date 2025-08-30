from fastapi import FastAPI
import logging

# Import your router
from src.routers import few_shot_router
from src.routers import tf_logistic_regression_router
from src.routers import llm_classifier_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

def create_app() -> FastAPI:
    """Create and configure FastAPI app"""
    app = FastAPI(
        title="Contract Classifier API",
        description="Contract classification service for contracts",
        version="1.0.0"
    )

    # Include routers
    app.include_router(few_shot_router.router)
    app.include_router(tf_logistic_regression_router.router)
    app.include_router(llm_classifier_router.router)

    @app.get("/")
    async def root():
        return {"message": "🚀 Contract Classifier API is running!"}

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
