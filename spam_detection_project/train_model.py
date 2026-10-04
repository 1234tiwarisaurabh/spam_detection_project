"""
train_model.py
Trains a Naive Bayes spam classifier (SMS/Email style text)
and saves model.pkl + vectorizer.pkl for the Flask app.

Run: python train_model.py
"""

import random
import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

random.seed(42)

# ---------------------------------------------------------
# 1. BUILD A SYNTHETIC BUT REALISTIC SPAM/HAM DATASET
# ---------------------------------------------------------

spam_templates = [
    "Congratulations! You have WON a {prize} worth ${amount}. Claim now at {link}",
    "URGENT: Your account will be suspended. Click {link} to verify immediately",
    "You are selected for a FREE {prize}! Reply YES to claim before it expires",
    "Limited time offer! Get {percent}% OFF on all products. Visit {link} now",
    "Dear customer, you have an unclaimed prize of ${amount}. Call {phone} now",
    "WINNER!! As a valued network customer you have been selected to receive a ${amount} prize",
    "Your loan of ${amount} has been approved. No credit check needed. Apply at {link}",
    "Hot singles in your area are waiting to chat with you. Click {link}",
    "FREE entry into our {amount} weekly draw, text WIN to {phone} now",
    "You have been chosen to receive a free {prize}. Confirm your address at {link}",
    "Click here to claim your free {prize} before the offer ends today {link}",
    "Congratulations, your number has won ${amount} in the mobile lottery. Contact {phone}",
    "Get rich quick! Invest ${amount} today and double it in 24 hours, visit {link}",
    "Your package could not be delivered, pay a fee of ${amount} at {link} to release it",
    "IRS Notice: You owe back taxes. Pay immediately at {link} to avoid arrest",
    "Act now! Only a few {prize} left, claim yours instantly at {link}",
    "You've been pre-approved for a ${amount} credit card, apply now at {link}",
    "Make ${amount} per week working from home, no experience needed, click {link}",
    "This is your final notice, your subscription payment of ${amount} failed, update at {link}",
    "Text STOP to unsubscribe or WIN to enter our ${amount} cash giveaway now",
]

ham_templates = [
    "Hey, are we still meeting for {activity} tomorrow at {time}?",
    "Can you send me the notes from today's {subject} class?",
    "Don't forget to submit the {subject} assignment before {time}",
    "I'll be a bit late for {activity}, see you around {time}",
    "Thanks for helping me with the {subject} project yesterday",
    "Let's grab {activity} this weekend, are you free?",
    "The {subject} exam has been postponed to next week",
    "Please review the code I pushed for the {subject} module",
    "Mom, I'll reach home by {time}, don't wait for dinner",
    "Reminder: team meeting about {subject} at {time} tomorrow",
    "Happy birthday! Hope you have a great {activity} today",
    "Can we reschedule our {activity} call to {time}?",
    "I finished the {subject} report, sending it to you now",
    "Are you coming to the {activity} event this evening?",
    "Just checking in, how did your {subject} interview go?",
    "Let's discuss the {subject} presentation before {time}",
    "I saved you a seat for {activity}, come join us",
    "Great job on the {subject} project, really impressed!",
    "See you at the {activity} practice at {time}",
    "Can you pick up groceries on your way back from {activity}?",
]

fill_prize = ["iPhone 16", "vacation package", "gift card", "laptop", "cash prize", "smartwatch", "PlayStation 5"]
fill_amount = ["500", "1000", "5000", "10000", "250", "2500", "750"]
fill_percent = ["50", "70", "80", "90", "60"]
fill_link = ["bit.ly/claim-now", "http://free-prize-win.com", "http://secure-bank-update.net", "http://claim-reward.xyz"]
fill_phone = ["09876543210", "18005551234", "07000123456"]
fill_activity = ["lunch", "the movie", "cricket", "study session", "coffee", "the gym", "badminton"]
fill_subject = ["Data Structures", "Machine Learning", "DBMS", "Operating Systems", "Web Dev", "Maths"]
fill_time = ["6 PM", "10 AM", "noon", "5:30 PM", "9 AM", "8 PM"]


def fill_template(t):
    return t.format(
        prize=random.choice(fill_prize),
        amount=random.choice(fill_amount),
        percent=random.choice(fill_percent),
        link=random.choice(fill_link),
        phone=random.choice(fill_phone),
        activity=random.choice(fill_activity),
        subject=random.choice(fill_subject),
        time=random.choice(fill_time),
    )


def build_dataset(n_per_class=350):
    rows = []
    for _ in range(n_per_class):
        t = random.choice(spam_templates)
        rows.append({"label": "spam", "text": fill_template(t)})
    for _ in range(n_per_class):
        t = random.choice(ham_templates)
        rows.append({"label": "ham", "text": fill_template(t)})
    df = pd.DataFrame(rows).drop_duplicates(subset="text").reset_index(drop=True)
    return df


def main():
    print("Building dataset...")
    df = build_dataset(n_per_class=400)
    print(f"Dataset size: {len(df)}  (spam={sum(df.label=='spam')}, ham={sum(df.label=='ham')})")

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/spam.csv", index=False)
    print("Saved dataset -> data/spam.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    print("Vectorizing (TF-IDF)...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Multinomial Naive Bayes...")
    model = MultinomialNB(alpha=0.3)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, pos_label="spam")
    rec = recall_score(y_test, preds, pos_label="spam")
    f1 = f1_score(y_test, preds, pos_label="spam")
    cm = confusion_matrix(y_test, preds, labels=["ham", "spam"])

    print("\n===== MODEL EVALUATION =====")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall   : {rec*100:.2f}%")
    print(f"F1 Score : {f1*100:.2f}%")
    print("Confusion Matrix [ham, spam]:")
    print(cm)

    os.makedirs("model", exist_ok=True)
    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("model/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # save metrics for display in the web UI
    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    with open("model/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    print("\nSaved model -> model/model.pkl")
    print("Saved vectorizer -> model/vectorizer.pkl")
    print("Done. Now run: python app.py")


if __name__ == "__main__":
    main()
