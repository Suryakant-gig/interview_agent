from fastapi import FastAPI
from app.routes import router

# Create FastAPI app
app = FastAPI(
    title="AI Interview Agent",
    description="Offline RAG-based interview evaluation system",
    version="1.0.0"
)

# Health check (important for production)
@app.get("/")
def health_check():
    return {"status": "running"}

app.include_router(router)