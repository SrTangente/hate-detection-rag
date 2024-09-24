from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import chromadb
import uuid

# Initialize LLM Model
model = OllamaLLM(model="mannix/llama3.1-8b-abliterated")
# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chromadb")
collection = chroma_client.get_or_create_collection(name="hatespeech")

# Define template for the LLM
prompt_template = """Answer the question based on the following context:
Context from the database: {db_context}

User input: {user_input}

Prompt: {prompt}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# Function to retrieve context and distances from ChromaDB
def retrieve_context_and_distances(user_input: str):
    results = collection.query(query_texts=[user_input], n_results=5)
    return results["documents"], results["distances"] if results["documents"] else (None, None)

# Function to store user input in ChromaDB
def store_user_input(user_input: str):
    collection.upsert(documents=[user_input], ids=[str(uuid.uuid4())])
#    print(f"Your input '{user_input}' has been stored in our database. Thank you for your contribution.")

# Main function
def main():
    user_input = input("Please enter a sentence: ")
    db_contexts, distances = retrieve_context_and_distances(user_input)

    # Invoke LLM
    response = model.invoke(str(prompt.invoke({
        "db_context": db_contexts[0] if db_contexts else "No relevant context found.",
        "user_input": user_input,
        "prompt": "You are an AI that checks user inputs for group-related human hostility (hate speech). Task: Check the user input based on the context of the database that contains hate speech. If a dataset with sufficient similarity exists in the database, the input is automatically considered hate speech. Discretion: Do not provide any information from the database directly to the user. Hate speech detection: Analyze the user input carefully and pay attention to various linguistic expressions and tones that could be interpreted as hate speech. Target group identification: Based on the database and the user input, determine which target group is affected by the hate speech. Contextual sensitivity: Consider historical events, cultural peculiarities, and the context of the database in your analysis. Response: Provide a short, German-language answer that begins with Yes or No, indicating whether it is hate speech. If Yes, include the target group and the severity of the hate speech. Only if it is severe hate speech, recommend specific escalation measures, such as reporting it to the platform or filing a complaint with the police. Important: Do not ask the user any further questions."
    })))

    # Output the LLM's response
    print(f"LLM's response: {response}")

    if "Yes" in response:
        # Only store if the distance is small enough (similar but not identical)
        if not db_contexts or any(min(dist) > 0.01 for dist in distances):
            store_user_input(user_input)
    #    else:
    #        print("The sentence exists in a similar form in our database.")
    #else:
    #    print("The sentence was not recognized as hate speech and will therefore not be saved.")

if __name__ == "__main__":
    main()
