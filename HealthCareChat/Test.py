import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import torch
from transformers import GPTNeoForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import streamlit as st

import checkIntent as ci
import TFidf as tfidf


#patient_query = "tell me Claim Status?"





##### PRE TRAINED MODEL START #############



# # Load GPT Neo Model
# ##Load GPT-Neo model and tokenizer
        
# model_name = "EleutherAI/gpt-neo-125M"  # You can choose a larger model like gpt-neo-1.3B if needed
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = GPTNeoForCausalLM.from_pretrained(model_name)

# ## Train With Few Shot Learning
# # Create a prompt with a few examples for in-context learning
# def generate_prompt(user_input):
#     examples = ""
#     for i in range(min(len(df), 3)):  # Include 3 examples from the dataframe
#         examples += f"Q: {df['query'].iloc[i]}\nA: {df['response'].iloc[i]}\n\n"

#     # Combine examples with the user's question
#     prompt = examples + f"Q: {user_input}\nA:"
#     return prompt

# # Function to generate response using GPT-Neo
# def gpt_neo_response(user_input):
#     prompt = generate_prompt(user_input)
#     inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

#     # Generate response
#     output = model.generate(
#         **inputs,
#         max_new_tokens=50,
#         pad_token_id=tokenizer.eos_token_id
#     )

#     # Decode the generated response
#     response = tokenizer.decode(output[0], skip_special_tokens=True)
#     # Extract only the answer from the response (after the user's query)
#     response = response.split("A:")[-1].strip()
#     return response


# # # Example usage
# # user_input = "What is status of file 705623?"
# # response = gpt_neo_response(user_input)
# # print(response)


## Fine Tune Model on Data
# Convert data to the desired text format
## This code is to be executed only Once to generate Tokenizer and Model


######## TRAINING STARTS ################

# with open('./pickle_files/query_response_data.pkl', 'rb') as f:
#     loaded_query_response_data = pickle.load(f)

# listofQuestion = loaded_query_response_data


# with open("query_response_dataset.txt", "w") as file:
#     for item in listofQuestion:
#         query = item['query']
#         response = item['response']
#         file.write(f"Q: {query}\nA: {response}\n<|endoftext|>\n")

# # Function to load and tokenize dataset
# def load_dataset(file_path, tokenizer, block_size=512):
#     dataset = TextDataset(
#         tokenizer=tokenizer,
#         file_path=file_path,
#         block_size=block_size
#     )
#     return dataset

# # Prepare data collator
# def prepare_collator(tokenizer):
#     return DataCollatorForLanguageModeling(
#         tokenizer=tokenizer, 
#         mlm=False  # GPT-Neo does causal (autoregressive) language modeling, so MLM is False
#     )

# # Load the dataset (assuming your data is in a text file)
# train_dataset = load_dataset("./query_response_dataset.txt", tokenizer)


# # Prepare the data collator
# data_collator = prepare_collator(tokenizer)

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir="./gpt_neo_finetuned",
#     overwrite_output_dir=True,
#     num_train_epochs=3,          # You can adjust the number of epochs
#     per_device_train_batch_size=2,
#     save_steps=500,              # Save checkpoint every 500 steps
#     save_total_limit=2,          # Only keep 2 last checkpoints
#     logging_dir="./logs",        # Directory for logging
#     logging_steps=10,            # Log every 10 steps
#     learning_rate=5e-5,          # Fine-tuning learning rate
#     weight_decay=0.01,           # Weight decay for regularization
# )

# # Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     data_collator=data_collator,
# )

# # Start the fine-tuning process
# ### Run This Step only Once
# # trainer.train()


# # Save the fine-tuned model and tokenizer
# # trainer.save_model("./gpt_neo_finetuned_model")
# # tokenizer.save_pretrained("./gpt_neo_finetuned_tokenizer")

############## TRAINING ENDS #########################

# Load the fine-tuned model and tokenizer
model_path = "./gpt_neo_finetuned_model"  # Path to your fine-tuned model
tokenizer_path = "./gpt_neo_finetuned_tokenizer"  # Path to your fine-tuned tokenizer

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = GPTNeoForCausalLM.from_pretrained(model_path)


# Set the padding token to eos_token (or add a new pad_token if desired)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token
    # Alternatively, you can define a new padding token
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Ensure the model knows about the new special token if one was added
model.resize_token_embeddings(len(tokenizer))

# Function to generate a prompt for the model
def generate_prompt(user_input):
    # Customize the prompt to follow your query-response format
    prompt = f"Q: {user_input}\nA:"
    return prompt

# Function to check if response is valid or needs fallback
def is_valid_response(response):
    # Check if the response is long enough and doesn't just repeat the question
    min_length = 10  # Minimum length for a valid response
    if len(response) < min_length:
        return False
    
    # You can add other heuristics, e.g., if response contains certain phrases, return False
    if "I don't know" in response or response.strip() == "":
        return False
    
    return True


# Function to generate response using the fine-tuned GPT-Neo model
def gpt_neo_response_finetune(user_input, max_new_tokens=50, temperature=0.7, top_p=0.9, top_k=50):
    # Prepare the prompt
    prompt = generate_prompt(user_input)
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding=True)
    
    # Generate response
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,  # Control how many tokens are generated
        do_sample=True,                 # Use sampling for diverse responses
        temperature=temperature,        # Control randomness of the output
        top_p=top_p,                    # Nucleus sampling to focus on the top probability mass
        top_k=top_k,                    # Limits sampling to top-k tokens
        pad_token_id=tokenizer.pad_token_id,  # Use the defined padding token
    )
    
    # Decode the generated tokens into text
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract the relevant answer (anything after "A:")
    if "A:" in response:
        response = response.split("A:")[-1].strip()
    
    # Check if the response is valid, otherwise return a fallback message
    if not is_valid_response(response):
        response = "I'm sorry, I didn't understand that. Could you please rephrase your question?"


    return response


# # Example usage
# user_input = "What is the status of my claim with file number 7057039?"
# response = gpt_neo_response_finetune(user_input)
# print(response)




##### STREAMLIT STARTS ############






# Set up the Streamlit app
st.title("Healthcare Claim Virtual Agent")
st.write("Ask questions related to your healthcare claims.")


# Input for the patient's question
patient_query = st.text_input("Enter your query about your claim:")

# If user submits a query
if st.button("Submit"):
    ## Identify the Intent of the Question 
    response,failsafe = tfidf.runNLPchat(patient_query)
    print(response,failsafe)
    if failsafe:
        st.write(f"Response: {response}")
    else:
        response = gpt_neo_response_finetune(patient_query)
        st.write(f"Response: {response}")



