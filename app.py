import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.chains import LLMChain


load_dotenv()

#This function reads the data
def load_data(master_data, retailer_data):
    master_df = pd.read_csv("master_southern_glazer.csv", encoding='latin1')
    retail_df = pd.read_csv("Retailer_Beers_Data.csv")
    return  master_df, retail_df

#Creating the Agent here combining the prompt and llm.
def similarity_search_agent():

    llm = OpenAI(temperature=0)

    prompt = PromptTemplate(
        input_variables=['master_df', 'retailer_df'],
        template="""
        You are an AI Agent that matches records from retailer dataset to master dataset
        
        Task:
        - For each record in retailer dataset, find best matching record in master dataset based on similarity across all the columns.
        - Use appropriate methods (like cosine similarity search) to handle differences in data.
        - Also its your task to understand how exactly other fields are matched based on units and quantity or other context example: conversion of oz to litres or Milli-litres 
        - Combine the retailer record and the matched master record into a single dataset.
        - Collect all matched records in single dataset.
        - Save the new dataset to a csv file "matched_results.csv".
        
        Instructions:
        - Work systematically through the retailer dataset.
        - Ensure that the matches are accurate and the best possible.
        - Do not provide analysis or insights; just perform matching task.
        
        output:
        - confirm once the task is completed and file is saved.
        
        Begin the task now.      
        """
    )

    #chaining the llm and prompt
    chain = prompt | llm

    return chain


def cosine_Similarity_search(master_df, retailer_df, threshold):
    matches = []

    #combining the text columns into single string for each row.
    master_df['combined_text'] = master_df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    retailer_df['combined_text'] = retailer_df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)


    # TF-IDF vectors
    vectorizer = TfidfVectorizer().fit(pd.concat([master_df['combined_text'], retailer_df['combined_text']]))
    master_vectors = vectorizer.transform(master_df['combined_text'])
    retailer_vectors = vectorizer.transform(retailer_df['combined_text'])

    for idx, retailer_vector in enumerate(retailer_vectors):
        best_match = None
        highest_Score =0





        #cosine_similarity
        cos_similarity = cosine_similarity(retailer_vector, master_vectors).flatten()

        # find best mactch

        if cos_similarity.max() >= threshold:
            highest_Score = cos_similarity.max()
            best_match_idx= cos_similarity.argmax()
            best_match = master_df.iloc[best_match_idx]



        if best_match is not None:
            combined_record = retailer_df.iloc[idx].to_dict()

            for key, value in best_match.items():
                combined_record[f"master_{key}"] = value

            combined_record['similarity_score'] = highest_Score

            matches.append(combined_record)

    print(f"Total matches found: {len(matches)}")
    if matches:
        print(pd.DataFrame(matches).head())  # Show first few matched records

    return pd.DataFrame(matches)

def automate_process(master_file, retailer_file):
    # Load data
    master_df, retailer_df = load_data(master_file, retailer_file)

    # Initialize the agent
    agent = similarity_search_agent()

    # Perform the matching using cosine similarity
    threshold = 0.1  # Adjust the threshold as needed (0-1 scale for cosine similarity)
    matched_df = cosine_Similarity_search(master_df, retailer_df, threshold)

    # Save the results
    output_file = "matched_results.csv"
    matched_df.to_csv(output_file, index=False)
    print(f"Matched records saved to {output_file}")

    # Use the invoke method to run the agent
    inputs = {'master_df': str(master_df.head()), 'retail_df': str(retailer_df.head())}
    confirmation = agent.invoke(inputs)
    print(confirmation)

if __name__ == "__main__":
    master_file = "master_southern_glazer.csv"
    retailer_file = "Retailer_Beers_Data.csv"
    automate_process(master_file, retailer_file)







