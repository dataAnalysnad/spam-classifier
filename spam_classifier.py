import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords

# Download stopwords (only the first time)
nltk.download('stopwords')

# Step 1: Load the dataset from spam.csv
df = pd.read_csv('spam.csv')

# Check the first few rows to understand the dataset
print("\n========== Dataset Preview ==========")
print(df.head())
print("=====================================\n")

# Step 2: Text Preprocessing
# Remove stopwords and convert text to lowercase
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()  # Lowercase the text
    words = text.split()  # Tokenize the text (split by spaces)
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

# Apply preprocessing on the 'Message' column
df['Message'] = df['Message'].apply(preprocess_text)

# Step 3: Feature Extraction using Bag of Words (BoW)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Message'])  # Convert text to numerical features

# Step 4: Encode Labels (ham = 0, spam = 1)
y = df['Category'].map({'ham': 0, 'spam': 1})  # Convert labels to binary (0 for ham, 1 for spam)

# Step 5: Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the Model
print("\n========== Model Evaluation ==========")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("======================================\n")

# Step 9: Allow user to input a message for spam detection
print("\n========== Spam Classifier ==========")
print("Type a message to classify it as SPAM or HAM.")
print("Enter 'exit' to quit.\n")

while True:
    user_message = input("Enter a message: ")
    if user_message.lower() == 'exit':
        print("\nExiting spam classifier. Goodbye!")
        break
    
    # Preprocess the user input
    processed_message = preprocess_text(user_message)
    # Transform the input using the same vectorizer
    transformed_message = vectorizer.transform([processed_message])
    # Predict using the trained model
    prediction = model.predict(transformed_message)
    # Output the result
    result = "SPAM" if prediction[0] == 1 else "HAM"
    print(f"\nThe message is classified as: {result}")
    print("======================================\n")