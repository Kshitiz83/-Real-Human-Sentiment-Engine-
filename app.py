from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import xgboost as xgb
import scipy.sparse as sp
import numpy as np

app = FastAPI(title="Real Human Sentiment Engine")

# Load our serialized preprocessing layers safely via joblib
tfidf = joblib.load('tfidf_vectorizer.joblib')
ohe = joblib.load('airline_encoder.joblib')

# STABILIZED LOADING: Use native core Booster to completely prevent wrapper TypeErrors
bst = xgb.Booster()
bst.load_model('xgb_sentiment_model.json')

class_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}

class TweetPayload(BaseModel):
    text: str
    airline: str
    retweet_count: int

@app.get("/")
def health_check():
    return {"status": "online", "model": "XGBoost Text-Tabular Hybrid Native Booster"}

@app.post("/predict")
def predict_sentiment(payload: TweetPayload):
    # Transform textual input
    text_sparse = tfidf.transform([payload.text])
    
    # Transform categorical input
    ohe_sparse = ohe.transform([[payload.airline]])
    
    # Build standard numeric input
    numeric_sparse = sp.csr_matrix([[payload.retweet_count]])

    # Reconstruct identical horizontal structural alignment pipeline
    X_inference = sp.hstack([text_sparse, ohe_sparse, numeric_sparse], format='csr')

    # Convert sparse matrix into optimized internal DMatrix required by Booster
    dmatrix_inference = xgb.DMatrix(X_inference)

    # Predict probability distribution array natively 
    prob_distribution = bst.predict(dmatrix_inference)
    predicted_class_id = int(np.argmax(prob_distribution[0]))

    return {
        "predicted_sentiment": class_mapping[predicted_class_id],
        "confidence_scores": {
            "negative": float(prob_distribution[0][0]),
            "neutral": float(prob_distribution[0][1]),
            "positive": float(prob_distribution[0][2])
        }     
    }
