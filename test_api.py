"""
Test the API.
=============
With the server running in one terminal (uvicorn app:app --reload),
run this in another terminal to check every endpoint works.

Run with:   python test_api.py
"""

import requests

BASE = "http://127.0.0.1:8000"

print("1. Health check")
r = requests.get(f"{BASE}/health")
print("  ", r.status_code, r.json())

print("\n2. Weak applicant (expect DECLINE)")
r = requests.post(f"{BASE}/predict", json={
    "age": 29, "annual_income": 52000, "years_employed": 3.0,
    "credit_score": 610, "loan_amount": 48000, "loan_term_months": 60,
    "num_credit_cards": 5, "debt_to_income": 0.923,
})
print("  ", r.status_code, r.json())

print("\n3. Strong applicant (expect APPROVE)")
r = requests.post(f"{BASE}/predict", json={
    "age": 45, "annual_income": 95000, "years_employed": 20.0,
    "credit_score": 780, "loan_amount": 30000, "loan_term_months": 24,
    "num_credit_cards": 2, "debt_to_income": 0.31,
})
print("  ", r.status_code, r.json())

print("\n4. Bad input - impossible credit score (expect 422 rejection)")
r = requests.post(f"{BASE}/predict", json={
    "age": 29, "annual_income": 52000, "years_employed": 3.0,
    "credit_score": 9000, "loan_amount": 48000, "loan_term_months": 60,
    "num_credit_cards": 5, "debt_to_income": 0.923,
})
print("  ", r.status_code, "- rejected as expected" if r.status_code == 422 else "- UNEXPECTED")
