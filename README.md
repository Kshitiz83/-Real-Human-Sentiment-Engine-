# Multi-Modal Production Sentiment Engine: From Leak Detection to Containerized Cloud Microservice

A production-grade machine learning microservice that processes multi-modal (tabular + unstructured text) human feedback data. The project features a strict, leak-proof engineering pipeline, memory-efficient sparse matrix operations, and a low-level native deployment architecture hosted as a containerized API on **Render**.

---

## 🏗️ Architecture & Engineering Blueprint

---

## 🚀 Key Engineering Triumphs & Post-Mortem

### 1. Diagnostic Leak Detection & Dataset Pivot
* **The Problem:** Initial validation evaluations on a synthetic Kaggle dataset yielded suspiciously perfect metrics. 
* **The Audit:** Audited internal tree split coefficients using `xgb_clf.feature_importances_`. Discovered that a behavioral flag (`complaint_registered`) and deterministic keyword templates introduced severe data leakage.
* **The Fix:** Completely discarded the rigged synthetic dataset and pivoted to an authentic, chaotic human-written dataset (**Twitter US Airline Sentiment**).

### 2. Leak-Proof Multi-Modal Pipeline Design
* **Order of Operations:** Enforced a strict sequencing boundary: Target Mapping ──> Extraction ──> Train-Test Splitting ──> Vectorization/Encoding.
* **Array Enforcement:** Structured non-linear categorical encoding utilizing double brackets (`X_train_raw[['airline']]`) to satisfy Scikit-Learn's 2D input array specifications.
* **Vector Leak Isolation:** Restricted `TfidfVectorizer` (extracting unigrams and bigrams) to execute `.fit_transform()` exclusively on training rows, while applying `.transform()` to testing rows.

### 3. Sparse Array Stacking & Memory Efficiency
* **The Problem:** High-cardinality text vectors combined with one-hot encoded categories threatened to trigger massive RAM bloat and system memory deadlocks.
* **The Solution:** Combined all structural data elements horizontally into compressed sparse rows using `scipy.sparse.hstack(..., format='csr')`. This maintained a highly optimized memory footprint and prevented execution container crashing.

### 4. Multiprocessing Deadlock Resolution
* **The Problem:** Parallel processing configurations (`n_jobs=-1`) during K-Fold validation loops caused low-level system deadlocks when parsing sparse coordinate structures.
* **The Solution:** Refactored to a high-speed sub-validation split strategy monitored via `optuna.logging.INFO`. Found tight regularization thresholds that stabilized validation accuracy at an authentic **75.63%**.

### 5. Low-Level Production Serialization & Cloud Containerization
* **Dependency Pinning:** Resolved Render container build failures by explicitly locking down `PYTHON_VERSION=3.11` to compile modern `scikit-learn==1.9.0` resources without library mismatch issues.
* **Framework Bug Workaround:** Bypassed high-level framework initialization bugs (`TypeError: _estimator_type undefined`) inside the API layer by dropping high-level abstractions and loading weights natively into the low-level engine via `xgb.Booster()`.
* **State Serialization:** Separated structural components into an efficient production setup: Scikit-Learn transformers serialized via `.joblib`, and tree layout nodes saved natively as text-based `.json` layout weights.

---

## 🛠️ Tech Stack & Dependencies

* **Core ML Engine:** XGBoost (`xgb.Booster`)
* **Data & Matrix Engineering:** Scikit-Learn (v1.9.0), SciPy (Compressed Sparse Row layout), Pandas
* **Optimization Framework:** Optuna
* **Deployment & Container Infrastructure:** Python 3.11, Flask/FastAPI, Render Cloud Containers

---

## 📦 API Local Verification & Deployment

To launch the microservice locally using the core, low-level engine:

```bash
# Clone the repository
git clone https://github.com

cd sentiment-production-engine

# Build and run the service
pip install -r requirements.txt
python app.py
```

### Sample Payload Request
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"airline": "United", "text": "Flight was delayed 4 hours with zero communication. Horrible."}'
```
### Expected Response
```json
{
  "status": 200,
  "sentiment": "negative",
  "confidence": 0.9421
}
