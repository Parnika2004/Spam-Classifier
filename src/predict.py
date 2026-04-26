import joblib
from preprocess import clean_text

def predict(text: str) -> dict:
    model = joblib.load('models/classifier.pkl')
    
    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]
    
    return {
        'label': 'spam' if prediction == 1 else 'ham',
        'confidence': round(proba[prediction] * 100, 2)
    }

# Test it
if __name__ == '__main__':
    test = "Congratulations! You won a FREE prize. Call now!"
    result = predict(test)
    print(f"Prediction: {result['label']} ({result['confidence']}% confidence)")