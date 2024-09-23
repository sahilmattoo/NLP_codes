# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample Data
data = {
    'Query': [
        "Do you have access to the status of Coverage?","Do you have information of the Coverage?", "Can you provide me with the current status of my Coverage?",
        "Is it possible to check the status of my Coverage?", "Are you able to access the latest information on Coverage?",
        "Could you tell me if the Coverage status is available?","Do you know how to retrieve the status of my Coverage?",
        "provide Coverage status", "Help with Coverage Details","provide Coverage status", "Help with Coverage Details","provide Coverage status", "Help with Coverage Details",
        
        "Can you clarify why there has been a delay in processing the claim?", "What is the reason behind the delay of my claim?",
        "Can you help me with Claim?", "Can you help me understand the factors causing the claim delay?",
        "What reasons have been cited for the delay in my claim?",
        "provide Claim status", "Help with Claim Details","provide Claim status", "Help with Claim Details","provide Claim status", "Help with Claim Details",


        "ajshfkdjs","uedhkkedwe","euwiuefti", "asjdbcsd","ajshfkdjs","uedhkkedwe","euwiuefti", "asjdbcsd","ajshfkdjs","uedhkkedwe","euwiuefti", "asjdbcsd",
],
    'Intent':[
        "Coverage","Coverage","Coverage","Coverage","Coverage","Coverage","Coverage",
        "Coverage","Coverage","Coverage","Coverage","Coverage","Coverage",
        "Claim","Claim","Claim","Claim","Claim","Claim","Claim","Claim","Claim","Claim","Claim",
        "unknown","unknown","unknown","unknown","unknown","unknown","unknown","unknown","unknown","unknown","unknown","unknown",

]
    
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 1: Preprocessing the data
# Split data into features (X) and target (y)
X = df['Query']
y = df['Intent']

# Vectorize the Query column using TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(2,2))
X_tfidf = vectorizer.fit_transform(X)

# Encode the Intent labels into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 2: Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42)

# Step 3: Build and train the model (using Naive Bayes classifier)
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = classifier.predict(X_test)



def checkIntent(query):
    # Transform the input query using the trained TF-IDF vectorizer
    input_tfidf = vectorizer.transform([query])

    # Predict the intent using the trained model
    predicted_label_encoded = classifier.predict(input_tfidf)

    # Decode the predicted label back to the original intent
    predicted_intent = label_encoder.inverse_transform(predicted_label_encoded)
    return predicted_intent[0]




# # Sample input query
# input_query = "Help with Coverage Details"
# it = checkIntent(input_query)
# print(it)
# responsedictionary = {"Coverage": "System cannot provide coverage details without File Number",
#                       "Claim": "System cannot provide claim details without File Number",
#                       "unknown":"I'm sorry, I didn't understand that. Could you please rephrase your question?",}

# print(responsedictionary[it])

# # Sample input query
# input_query = "Can you help me with Claim?"
# it = checkIntent(input_query)
# print(it)
# responsedictionary = {"Coverage": "System cannot provide coverage details without File Number",
#                       "Claim": "System cannot provide claim details without File Number",
#                       "unknown":"I'm sorry, I didn't understand that. Could you please rephrase your question?",}


# print(responsedictionary[it])

# # Sample input query
# input_query = "ajshfkdjs?"
# it = checkIntent(input_query)
# print(it)
# responsedictionary = {"Coverage": "System cannot provide coverage details without File Number",
#                       "Claim": "System cannot provide claim details without File Number",
#                       "unknown":"I'm sorry, I didn't understand that. Could you please rephrase your question?",}

# print(responsedictionary[it])