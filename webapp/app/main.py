from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import chromadb
import uuid

# Initialize LLM Model
def initialize_model():
    try:
        model = OllamaLLM(model="mannix/llama3.1-8b-abliterated")
        return model
    except Exception as e:
        print(f"Error initializing LLM Model: {e}")
        return None

# Initialize ChromaDB
def initialize_chromadb():
    try:
        chroma_client = chromadb.PersistentClient(path="./chromadb")
        collection = chroma_client.get_or_create_collection(name="hatespeech")
        return collection
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        return None

# Define template for the LLM
def initialize_prompt_template():
    prompt_template = """Answer the question based on the following context: 
    Context from the database: {db_context}

    User input: {user_input}

    Prompt: {prompt}
    """
    try:
        prompt = ChatPromptTemplate.from_template(prompt_template)
        return prompt
    except Exception as e:
        print(f"Error creating prompt template: {e}")
        return None

# Function to retrieve context and distances from ChromaDB
def retrieve_context_and_distances(user_input: str, collection):
    if collection:
        try:
            results = collection.query(query_texts=[user_input], n_results=1)
            return results["documents"], results["distances"] if results["documents"] else (None, None)
        except Exception as e:
            print(f"Error retrieving context and distances: {e}")
    return None, None

# Function to store user input in ChromaDB
def store_user_input(user_input: str, collection):
    if collection:
        try:
            collection.upsert(documents=[user_input], ids=[str(uuid.uuid4())])
        except Exception as e:
            print(f"Error storing user input: {e}")

# Function to process the user input and generate a response
def process_user_input(user_input: str, model, prompt, collection):
    db_contexts, distances = retrieve_context_and_distances(user_input, collection)

    try:
        if model and prompt:
            # Invoke LLM
            response = model.invoke(str(prompt.invoke({
                "db_context": db_contexts[0] if db_contexts else "Kein relevanter Kontext gefunden.",
                "user_input": user_input,
                "prompt": "You are an AI that checks user input for group-related human hostility (hate speech). Task: Check the user input based on the context of the database that contains hate speech. If a dataset with sufficient similarity exists in the database, the input is automatically considered hate speech. Discretion: Do not provide any information from the database directly to the user. Hate speech detection: Analyze the user input carefully and pay attention to various linguistic expressions and tones that could be interpreted as hate speech. Target group identification: Based on the database and the user input, determine which target group is affected by the hate speech. Contextual sensitivity: Consider historical events, cultural peculiarities, and the context of the database in your analysis. Response: Provide a short, German-language answer that begins with Yes or No, indicating whether it is hate speech. If Yes, include the target group and the severity of the hate speech. Only if it is severe hate speech, recommend specific escalation measures, such as reporting it to the platform or filing a complaint with the police. Important: Do not ask the user any further questions."
            })))
        else:
            response = "Model or prompt template not initialized."
    except Exception as e:
        print(f"Error invoking LLM: {e}")
        response = "There was an error processing your request."

    if "Ja" in response:
        # Check if distances is not empty before using min()
        if not db_contexts or not distances or all(len(dist) == 0 for dist in distances) or any(dist and min(dist) > 0.01 for dist in distances):
            store_user_input(user_input, collection)

    return response
