import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import checkIntent as ci

# Read Data
data = pd.read_csv("./subset_df_2.csv")

# Transform Data
# Convert 'Company', 'Coverage', 'SubCoverage', 'Reason', 'SubReason', 'Disposition', 'Conclusion', 'Status' to categorical
for column in ['Company', 'Coverage', 'SubCoverage', 'Reason', 'SubReason', 'Disposition', 'Conclusion', 'Status']:
  data[column] = pd.Categorical(data[column])

# Convert 'Opened' and 'Closed' to datetime
data['Opened'] = pd.to_datetime(data['Opened'], errors='coerce')
data['Closed'] = pd.to_datetime(data['Closed'], errors='coerce')

# Convert 'Recovery' to float
data['Recovery'] = pd.to_numeric(data['Recovery'], errors='coerce')


# Change Data Format to Query-Response
claim_data = data.copy()

def formatData(claim_data):
  # Generate query-response pairs
    query_response_data = []

    for idx, row in claim_data.iterrows():
        # Query 1: Claim status
        query_response_data.append({
            'query': f"What is the status of my claim with file number {row['File No.']}?",
            'response': f"Your claim with File No. {row['File No.']} is currently {row['Status']}."
        })

        # # Query 2: Recovery amount
        # query_response_data.append({
        #     'query': f"What was the recovery amount for claim number {row['File No.']}?",
        #     'response': f"The recovery amount for your claim with File No. {row['File No.']} is ${row['Recovery']}."
        # })

        # Query 3: Denial reason (if applicable)
        if row['Disposition'] == 'Claim Settled':
            query_response_data.append({
                'query': f"Why was my claim with file number {row['File No.']} Claim Settled?",
                'response': f"Your claim with File No. {row['File No.']} was Settled on  {row['Closed']}."
            })

        # Query 4: Claim open and close dates
        query_response_data.append({
            'query': f"When was my claim with file number {row['File No.']} opened and closed?",
            'response': f"Your claim with File No. {row['File No.']} was opened on {row['Opened']} and closed on {row['Closed']}."
        })

        # # Query 5: Disposition reason
        # query_response_data.append({
        #     'query': f"Why is my claim with file number {row['File No.']} {row['Disposition']}?",
        #     'response': f"Your claim with File No. {row['File No.']} was {row['Disposition']} due to {row['SubReason']}."
        # })

    # Convert to DataFrame
    query_response_df = pd.DataFrame(query_response_data)
        # Save the trained vectorizer as a pickle file
    with open('./pickle_files/query_response_data.pkl', 'wb') as f:
        pickle.dump(query_response_data, f)

    return query_response_df

query_response_df = formatData(claim_data)


# Convert data to a DataFrame
df = query_response_df
# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(ngram_range=(2,3))
# Fit and transform the 'query' column
tfidf_matrix = vectorizer.fit_transform(df['query'])

# Save the files
# Save the trained vectorizer as a pickle file
with open('./pickle_files/tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)
# Save the trained vectorizer as a pickle file
with open('./pickle_files/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
# Load the vectorizer back from the pickle file
with open('./pickle_files/vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)
# Load the vectorizer back from the pickle file
with open('./pickle_files/tfidf_matrix.pkl', 'rb') as f:
    loaded_tfidf_matrix = pickle.load(f)




# Function to find the best match for a user query
def chatbot_response(user_input, loaded_vectorizer, loaded_tfidf_matrix, df):

    vectorizer = loaded_vectorizer
    tfidf_matrix = loaded_tfidf_matrix
    # Transform the user input to the same vector space as the TF-IDF matrix
    user_input_tfidf = vectorizer.transform([user_input])

    # Compute cosine similarity between user input and all predefined queries
    cosine_similarities = cosine_similarity(user_input_tfidf, tfidf_matrix)

    # Get the index of the most similar query
    best_match_index = cosine_similarities.argmax()

    # Get the score of the best match
    best_match_score = cosine_similarities[0, best_match_index]

    # Define a threshold for how close the match should be
    threshold = 0.2  # You can tune this
    failsafe = False


    try:

        # If the best match is above the threshold, return the corresponding response
        if best_match_score >= threshold:
            return df['response'].iloc[best_match_index],failsafe
        
        else:
            it = ci.checkIntent(user_input)
            print(it)
            responsedictionary = {"Coverage": "System cannot provide coverage details without File Number",
                                "Claim": "System cannot provide claim details without File Number"}


            nrs = responsedictionary[it]
            failsafe = True
            return nrs, failsafe
            #return "I'm sorry, I didn't understand that. Could you please rephrase your question?"
    except:
        failsafe = True
        return "I'm sorry, I didn't understand that. Could you please rephrase your question?", failsafe
    # Chatbot loop to simulate a conversation


def runNLPchat(patient_query):

# print("Welcome to the healthcare claim chatbot! Type 'exit' to end the conversation.")
# user_input = input("You: ")
    user_input = patient_query

    response = chatbot_response(user_input,loaded_vectorizer, loaded_tfidf_matrix, df)
    return response


# patient_query = "tell me Claim Status?"
# print(runNLPchat(patient_query))