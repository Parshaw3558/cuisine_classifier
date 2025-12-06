import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings("ignore")

print("\n Step 1: Loading dataset...")
df = pd.read_csv("Dataset.csv")


print("...Cleaning dataset...")

df = df.drop_duplicates()
df = df.dropna(subset=["Cuisines"])
df = df.fillna("Unknown")

required_cols = ["Restaurant Name", "City", "Average Cost for two", "Cuisines"]
for col in required_cols:
    if col not in df.columns:
        raise Exception(f" Missing column: {col}")


print(" Step 3: Preparing text features...")

df["text_data"] = (
    df["Restaurant Name"].astype(str) + " " +
    df["City"].astype(str) + " " +
    df["Average Cost for two"].astype(str)
)

X = df["text_data"]
y = df["Cuisines"]


print(" Step 4: Encoding labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)


print(" Step 5: Vectorizing text (TF-IDF)...")
tfidf = TfidfVectorizer(
    max_features=2000,        
    ngram_range=(1, 2),       
    min_df=3                 
)

X_vec = tfidf.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y_encoded, test_size=0.2, random_state=42
)


print(" Step 6: Training SVM model (LinearSVC)...")
model = LinearSVC()

model.fit(X_train, y_train)


print(" Step 7: Evaluating model...\n")

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



print(" Step 8: Saving model files...")

joblib.dump(model, "cuisine_model.pkl", compress=3)
joblib.dump(tfidf, "vectorizer.pkl", compress=3)
joblib.dump(le, "label_encoder.pkl", compress=3)

print("\n Model Training Complete!")
print(" Saved Files:")
print(" - cuisine_model.pkl")
print(" - vectorizer.pkl")
print(" - label_encoder.pkl")
print("\n This model is 100% Memory-Safe for Streamlit.")
