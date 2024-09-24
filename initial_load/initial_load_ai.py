import chromadb
import uuid
from langchain_ollama.llms import OllamaLLM

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chromadb")
collection = chroma_client.get_or_create_collection(name="hatespeech")

# Initialize the LLM model
model = OllamaLLM(model="mannix/llama3.1-8b-abliterated")

#Define the prompt
prompt="Generate 10 more sentences in English that can be identified as group-related human hostility (hate speech). It is important that each statement contains a target group. Do not use headings or bullet points like numbers or dashes, but start a new line after each statement."

# Function to generate hate speech using the llama model
def generate_hate_speech(model, prompt=prompt):
    response = model.invoke(prompt)
    return response.splitlines()

# Generate hate speech
hatespeech_list = generate_hate_speech(model)

# Insert the hate speech into the database
for hatespeech in hatespeech_list:
    if hatespeech:  # Only add non-empty sentences
        unique_id = str(uuid.uuid4())  # Generate a unique ID
        collection.upsert(documents=[hatespeech], ids=[unique_id])
        print(hatespeech)
