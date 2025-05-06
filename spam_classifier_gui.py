import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords

# Download stopwords (only the first time)
nltk.download('stopwords')

# Load and preprocess the dataset
df = pd.read_csv('spam.csv')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['Message'] = df['Message'].apply(preprocess_text)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Message'])
y = df['Category'].map({'ham': 0, 'spam': 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Function to classify user input
def classify_message():
    user_message = entry.get()
    if not user_message.strip():
        messagebox.showerror("Error", "Please enter a message.")
        return
    processed_message = preprocess_text(user_message)
    transformed_message = vectorizer.transform([processed_message])
    prediction = model.predict(transformed_message)
    result = "SPAM" if prediction[0] == 1 else "HAM"
    messagebox.showinfo("Result", f"The message is classified as: {result}")

# Create the GUI
root = tk.Tk()
root.title("Spam Classifier")

label = tk.Label(root, text="Enter a message to classify:", font=("Arial", 12))
label.pack(pady=10)

entry = tk.Entry(root, width=50, font=("Arial", 12))
entry.pack(pady=10)

button = tk.Button(root, text="Classify", command=classify_message, font=("Arial", 12), bg="blue", fg="white")
button.pack(pady=10)

root.mainloop()