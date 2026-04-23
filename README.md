# ML Lead Scoring Model

Trains a logistic regression model on firmographic features to score ICP fit, then applies the model to real companies and patches firmographic scores (0-100) back to each company record in HubSpot.

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

**15-state cap on one-hot encoding.** Companies are located across many U.S. states, but one-hot encoding every state creates a sparse feature matrix where most states have too few training examples to learn meaningful coefficients. The top 15 states by company count are encoded individually. Everything else rolls up to an "other" bucket. This keeps the feature space compact and prevents overfitting on rare states.

**Scores surfaced as 0-100 rather than raw probabilities.** The model outputs a probability between 0 and 1 internally. On the scoring side, the probability is multiplied by 100 and displayed as an integer. Reps and RevOps people think in hundred-point scales, not decimal probabilities. "87" is more intuitive than "0.87," and the UX matters because the score is surfaced inside HubSpot where non-technical users interact with it.

**Training labels constructed from rules rather than outcomes.** Real closed-won outcomes would be the right training labels, but they weren't available in sufficient volume for this dataset. Instead, labels were constructed from heuristic rules encoding domain knowledge: companies with strong Medicare participation, relevant headcount, and revenue in the target range are labeled positive; companies failing those criteria are labeled negative. This is a bootstrapping approach. The model learns what an ideal customer looks like based on the rules, then scores new companies against that pattern. In production, labels should transition to actual closed-won outcomes as they accumulate, and the model retrained on that data.

**200-record training set sized to available data.** The model trains on 200 companies because that's what was in the HubSpot portal. This is small for machine learning, but sufficient to demonstrate the scoring logic and produce directionally useful scores. Production scale would involve thousands of labeled historical companies, ideally augmented with outcome data from closed deals.

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
