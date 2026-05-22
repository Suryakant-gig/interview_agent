from fastapi import FastAPI

from app.routes.upload import (
    router as upload_router
)

from app.routes.evaluation import (
    router as evaluation_router
)
from app.routes.interview import (
    router as interview_router
)


# ---------------------------------
# Create app
# ---------------------------------
app = FastAPI(
    title="Interview Agent"
)

# ---------------------------------
# Register upload routes
# ---------------------------------
app.include_router(
    upload_router
)

app.include_router(
    interview_router
)
# ---------------------------------
# Register evaluation routes
# ---------------------------------
app.include_router(
    evaluation_router
)

# ---------------------------------
# Root route
# ---------------------------------
@app.get("/")
def root():

    return {
        "message":
        "Interview Agent Running"
    }