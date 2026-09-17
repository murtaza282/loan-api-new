"""
Loan Default Prediction API
============================
A small web service that wraps our trained model, so any application can
send a customer's details and get back a default prediction.

Run locally with:   uvicorn app:app --reload
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ---------------------------------------------------------------------------
# 1. Load the trained model ONCE, when the service starts up.
#    We do NOT load it inside the prediction function - that would reload it
#    from disk on every single request, which is slow. Load once, reuse.
# ---------------------------------------------------------------------------
model = joblib.load("loan_pipeline.joblib")


# ---------------------------------------------------------------------------
# 2. Create the FastAPI application object.
#    'app' is the thing the web server runs. The title/description show up
#    in the automatic documentation FastAPI generates for us.
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Loan Default Prediction API",
    description="Send a customer's details, get back a default probability.",
    version="1.0",
)

app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_methods=["*"],
allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 3. Define the SHAPE of the incoming data.
#    This class says: "a valid request must have these 8 fields, with these
#    types and these limits." Pydantic checks every incoming request against
#    it automatically. If a field is missing, the wrong type, or out of
#    range, FastAPI rejects the request with a clear error - before our model
#    ever sees it. This is input validation, and it's a core part of any
#    real service: never trust the input.
# ---------------------------------------------------------------------------
class Applicant(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Age in years")
    annual_income: float = Field(..., ge=0, description="Annual income")
    years_employed: float = Field(..., ge=0, le=60, description="Years employed")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    loan_amount: float = Field(..., ge=0, description="Requested loan amount")
    loan_term_months: int = Field(..., ge=1, le=360, description="Loan term in months")
    num_credit_cards: int = Field(..., ge=0, le=50, description="Number of credit cards")
    debt_to_income: float = Field(..., ge=0, le=5, description="Debt-to-income ratio")

    # An example request, shown in the auto-generated docs so users see the
    # expected format immediately.
    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 29,
                "annual_income": 52000,
                "years_employed": 3.0,
                "credit_score": 610,
                "loan_amount": 48000,
                "loan_term_months": 60,
                "num_credit_cards": 5,
                "debt_to_income": 0.923,
            }
        }
    }


# ---------------------------------------------------------------------------
# 4. Define the SHAPE of what we send back.
#    Not strictly required, but it makes the response predictable and
#    self-documenting: always a probability and a decision.
# ---------------------------------------------------------------------------
class Prediction(BaseModel):
    default_probability: float
    decision: str


# ---------------------------------------------------------------------------
# 5. The health-check endpoint.
#    Every deployed service needs a simple "are you alive?" URL. Cloud
#    platforms ping it to know the service is running. It does nothing but
#    return OK - that's the point.
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 6. The prediction endpoint - the actual job of this service.
#    - @app.post("/predict") means: this runs when someone sends a POST
#      request to the /predict URL.
#    - 'applicant: Applicant' means FastAPI validates the incoming JSON
#      against our Applicant class first. If it's invalid, our code never
#      runs - the caller gets a clear error automatically.
#    - We convert the validated data into the one-row DataFrame our pipeline
#      expects, get the probability, apply the 0.5 threshold, and return
#      clean JSON.
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=Prediction)
def predict(applicant: Applicant):
    # Turn the incoming applicant into a single-row DataFrame with the exact
    # column names the model was trained on.
    data = pd.DataFrame([applicant.model_dump()])

    # The pipeline scales AND predicts internally - one call.
    probability = model.predict_proba(data)[0][1]

    decision = "DECLINE" if probability >= 0.5 else "APPROVE"

    return Prediction(
        default_probability=round(float(probability), 4),
        decision=decision,
    )


# ---------------------------------------------------------------------------
# 7. A friendly root endpoint, so visiting the base URL isn't a 404.
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return FileResponse("index.html")
