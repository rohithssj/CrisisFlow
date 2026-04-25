from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from inference import run_simulation
import uvicorn

# 1. Initialize FastAPI App
app = FastAPI(
    title="CrisisFlow AI Decision API",
    description="Production-ready API for emergency response decision making",
    version="1.0.0"
)

# 2. Enable CORS (Critical for Frontend Integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Custom Exception Handler for Validation Errors (422 -> 400)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid request parameters. Please check your input."}
    )

# 3. Define Input Schema with Validation
class IncidentRequest(BaseModel):
    type: str = Field("medical", description="Type of incident (medical, cyber, fire, flood)")
    severity: int = Field(..., gt=0, description="Severity of the incident (1-10)")
    wait_time: float = Field(..., ge=0, description="Time patient has been waiting in minutes")
    distance: float = Field(..., ge=0, description="Distance to the incident site")

# 4. Define Response Schema
class DecisionResponse(BaseModel):
    unit: str
    risk: str
    score: int
    reason: str

# 5. Core Decision Endpoint
@app.post("/decision", response_model=DecisionResponse)
async def get_decision(request: IncidentRequest):
    # 1. Basic Validation (Safety Layer)
    if request.severity <= 0:
        raise HTTPException(status_code=400, detail="Severity must be greater than 0")
    if request.wait_time < 0 or request.distance < 0:
        raise HTTPException(status_code=400, detail="Wait time and distance cannot be negative")

    try:
        # 2. Run Simulation
        data = request.model_dump()
        result = run_simulation(data)
        
        # 3. Validate Engine Output Structure
        required_keys = ["unit", "risk", "score", "reason"]
        if not result or not all(key in result for key in required_keys):
            raise ValueError("Incomplete engine output")
            
        return result
        
    except Exception as e:
        # 4. Production-Safe Generic Error
        raise HTTPException(
            status_code=500, 
            detail="Internal server error"
        )

# 6. Health Check Endpoint
@app.get("/")
async def root():
    return {"status": "CrisisFlow API is active", "engine": "FastAPI"}

# Start server locally
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
