"""
Product Sentiment Analyzer
---------------------------
Analyzes product reviews and classifies them as Positive, Negative, or Neutral
using a Naive Bayes classifier (scikit-learn) trained on a sample dataset.

Run:
    python sentiment_analyzer.py
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------
# 1. Sample training data (product reviews).
#    Replace this with your own CSV using load_from_csv() below.
# ---------------------------------------------------------------
data = {
    "review": [
        "This product is amazing, works perfectly and looks great!",
        "Excellent quality, exceeded my expectations, highly recommend.",
        "I love it, best purchase I have made this year.",
        "Great value for money, very satisfied with the build.",
        "Superb performance, fast delivery, will buy again.",
        "Fantastic product, does exactly what it promises.",
        "Good product, works fine for daily use.",
        "Nice design and easy to use, happy with it.",
        "Outstanding camera quality and battery life, love it!",
        "Best purchase, the sound quality is incredible.",
        "Super comfortable and durable, worth every penny.",
        "Impressed by the fast charging and sleek design.",
        "Amazing customer support and product quality both.",
        "Highly satisfied, works better than expected.",
        "Perfect gift, everyone in the family loved it.",
        "Really happy with this, smooth performance every time.",

        "Terrible product, stopped working within a week.",
        "Very disappointed, quality is extremely poor.",
        "Waste of money, do not buy this.",
        "Worst purchase ever, completely broken on arrival.",
        "Bad experience, customer service was unhelpful too.",
        "Not worth the price, cheap materials used.",
        "Product broke after two days, very frustrating.",
        "Horrible quality control, item was defective.",
        "Charger stopped working after one use, total waste.",
        "Screen cracked on the first day, very poor build.",
        "Battery drains too fast, extremely disappointing.",
        "Received a damaged item, packaging was careless.",
        "Regret buying this, does not work as advertised.",
        "Poor performance and slow, would not recommend.",

        "The product is okay, nothing special about it.",
        "Average quality, does the job but not impressive.",
        "It's fine, meets basic expectations, nothing more.",
        "Neutral experience, product works as described.",
        "Decent product for the price, could be better.",
        "It arrived on time, packaging was normal.",
        "Standard quality, similar to other brands.",
        "Not bad, not great either, just average.",
        "Works okay for basic tasks, nothing extraordinary.",
        "It's an average blender, does what it says.",
        "Reasonable product, some features could improve.",
        "Fairly standard experience, no major complaints.",
        "It does the job, though design feels outdated.",
    ],
    "sentiment": [
        "Positive", "Positive", "Positive", "Positive", "Positive", "Positive", "Positive", "Positive",
        "Positive", "Positive", "Positive", "Positive", "Positive", "Positive", "Positive", "Positive",
        "Negative", "Negative", "Negative", "Negative", "Negative", "Negative", "Negative", "Negative",
        "Negative", "Negative", "Negative", "Negative", "Negative", "Negative",
        "Neutral", "Neutral", "Neutral", "Neutral", "Neutral", "Neutral", "Neutral", "Neutral",
        "Neutral", "Neutral", "Neutral", "Neutral", "Neutral",
    ],
}

df = pd.DataFrame(data)


def load_from_csv(path):
    """Load your own dataset. CSV must have columns: 'review', 'sentiment'."""
    return pd.read_csv(path)


# ---------------------------------------------------------------
# 2. Build and train the model
# ---------------------------------------------------------------
def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["sentiment"], test_size=0.25, random_state=42, stratify=df["sentiment"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("clf", MultinomialNB()),
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model


# ---------------------------------------------------------------
# 3. Predict sentiment for new reviews
# ---------------------------------------------------------------
def predict_sentiment(model, review_text):
    prediction = model.predict([review_text])[0]
    probs = model.predict_proba([review_text])[0]
    confidence = max(probs) * 100
    return prediction, confidence


# ---------------------------------------------------------------
# 4. Main program
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("Training sentiment analyzer...\n")
    model = train_model(df)

    print("\n--- Try it on new reviews ---\n")
    sample_reviews = [
        "This phone camera quality is outstanding, love the battery life too!",
        "The charger stopped working after one use, total waste of money.",
        "It's an average blender, works but nothing extraordinary.",
    ]

    for review in sample_reviews:
        sentiment, confidence = predict_sentiment(model, review)
        print(f"Review: {review}")
        print(f"Predicted Sentiment: {sentiment} (confidence: {confidence:.1f}%)\n")

    # Interactive mode
    print("--- Interactive mode (type 'exit' to quit) ---")
    while True:
        user_input = input("\nEnter a product review: ")
        if user_input.strip().lower() == "exit":
            break
        sentiment, confidence = predict_sentiment(model, user_input)
        print(f"Sentiment: {sentiment} (confidence: {confidence:.1f}%)")