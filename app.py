from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import xgboost as xgb
import scipy.sparse as sp
import numpy as np

app = FastAPI(title="Real Human Sentiment Engine")

# Load our serialized preprocessing and layout models offline
tfidf = joblib.load('tfidf_vectorizer.joblib')
ohe = joblib.load('airline_encoder.joblib')

# Initialize with parameters matching your multi-class target type
model = xgb.XGBClassifier(objective='multi:softprob', num_class=3)
model.load_model('xgb_sentiment_model.json')


class_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}

class TweetPayload(BaseModel):
    text: str
    airline: str
    retweet_count: int

@app.get("/")
def health_check():
    return {"status": "online", "model": "XGBoost Text-Tabular Hybrid"}

@app.post("/predict")
def predict_sentiment(payload: TweetPayload):
    text_sparse = tfidf.transform([payload.text])
    ohe_sparse = ohe.transform([[payload.airline]])
    numeric_sparse = sp.csr_matrix([[payload.retweet_count]])

    # Reconstruct identical matrix horizontal features pipeline layout
    X_inference = sp.hstack([text_sparse, ohe_sparse, numeric_sparse], format='csr')

    # Extract the raw 2D probability distribution array matrix
    prob_distribution = model.predict_proba(X_inference)
    predicted_class_id = int(np.argmax(prob_distribution))

    return {
        "predicted_sentiment": class_mapping[predicted_class_id],
        "confidence_scores": {
            # FIXED: Explicitly grab row index 0 to fetch the correct scalar probability values
            "negative": float(prob_distribution[0][0]),
            "neutral": float(prob_distribution[0][1]),
            "positive": float(prob_distribution[0][2])
        }     
    }
