# Lead Scoring Model

Trains a logistic regression model on synthetic health system data to predict which companies are ideal customer profiles (ICPs), then applies the model to real companies in HubSpot to assign firmographic scores (0-100). Scores are patched back to each company record in HubSpot. The model is trained on synthetic data with rule-based labels — it is fully functional end-to-end but would need real pipeline outcomes to produce production-grade predictions.

## Tech Stack

- **Language:** Python
- **Libraries:** scikit-learn (LogisticRegression, StandardScaler), pandas, requests, python-dotenv
- **API:** HubSpot CRM v3 (companies)
- **Model format:** pickle (model.pkl)

## How It Fits Into the GTM System

This is the scoring layer. It takes enriched company data (employee count, revenue, locations, medicare status) and outputs a firmographic score that feeds into the composite scoring dashboard. The composite score combines this firmographic score with engagement and pain signal scores to determine priority tier and rep routing.

## Key Design Decisions

- **Logistic regression over more complex models** — Binary classification (ICP vs. not) is the right framing for this problem, and logistic regression outputs calibrated probabilities that map cleanly to a 0-100 score. More complex models would overfit on 200 synthetic records.
- **Stratified synthetic data generation** — Training records are generated in three tiers (small, mid-size, large) with realistic ranges per tier, rather than uniform random across all ranges. This produces a more realistic distribution for the model to learn from.

## How to Run

**1. Install dependencies:**

```bash
pip install -r requirements.txt
```

**2. Generate training data (optional — CSV is already generated):**

```bash
python generate_training_data.py
```

This produces `fake_200_health_systems.csv` with 200 scored health system records.

**3. Train the model:**

```bash
python scoring_model.py
```

This reads the CSV, trains the model, and saves `model.pkl`.

**4. Score real HubSpot companies:**

Create a `.env` file:

```
HUBSPOT_API_KEY=pat-na1-your-token-here
```

Your Private App needs `crm.objects.companies.read` and `crm.objects.companies.write` scopes.

```bash
python score_real_companies.py
```

This pulls all companies from HubSpot, runs them through the trained model, and PATCHes a `firmographic_score` (0-100) back to each company record.

## Documentation

For feature selection rationale, scoring rules, known limitations, and future vision, see [scoring_writeup.md](scoring_writeup.md).

## Planned Extensions

- Train on real pipeline outcomes (closed-won vs. closed-lost) instead of synthetic rule-based labels
- Add model evaluation metrics (precision, recall, AUC) to the training output
- Expand state coverage beyond the 15 states used in training data
- Move from batch scoring to incremental scoring triggered by company property changes
- Add feature importance analysis to validate which firmographic signals actually predict conversion
# lead-scoring-model
