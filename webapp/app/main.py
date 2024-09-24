from langchain.prompts import ChatPromptTemplate
from langchain.llms import Ollama
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chains.base import Chain
from langchain.schema import Document
import uuid

# Initialize LLM Model
def initialize_model():
    try:
        model = Ollama(model="mannix/llama3.1-8b-abliterated")
        return model
    except Exception as e:
        print(f"Error initializing LLM Model: {e}")
        return None

# Initialize ChromaDB using Langchain
def initialize_chromadb():
    try:
        embedding_function = OllamaEmbeddings()
        chroma_db = Chroma(collection_name="hatespeech", embedding_function=embedding_function, persist_directory="./chromadb")
        return chroma_db
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

# Create a chain for retrieving context from ChromaDB
class ChromaRetrieveChain(Chain):
    def __init__(self, chroma_db):
        super().__init__()
        self.chroma_db = chroma_db

    def _call(self, inputs):
        user_input = inputs["user_input"]
        try:
            results = self.chroma_db.similarity_search(user_input, n_results=1)
            db_context = results[0].page_content if results else "No relevant context found."
            return {"db_context": db_context}
        except Exception as e:
            print(f"Error retrieving context from ChromaDB: {e}")
            return {"db_context": "Error retrieving context."}

    @property
    def input_keys(self):
        return ["user_input"]

    @property
    def output_keys(self):
        return ["db_context"]

# Store user input in ChromaDB
def store_user_input(user_input: str, chroma_db):
    try:
        chroma_db.add_texts(texts=[user_input], ids=[str(uuid.uuid4())])
    except Exception as e:
        print(f"Error storing user input: {e}")

# Create a chain for processing user input and generating a response
def create_processing_chain(model, prompt_template, chroma_db):
    # Step 1: Retrieve context from ChromaDB
    retrieval_chain = ChromaRetrieveChain(chroma_db)
    
    # Step 2: Create the LLM chain to generate a response
    llm_chain = LLMChain(
        llm=model, 
        prompt=prompt_template
    )
    
    # Combine the two chains sequentially
    full_chain = SimpleSequentialChain(chains=[retrieval_chain, llm_chain])
    return full_chain

# Function to process user input and generate a response
def process_user_input(user_input: str, processing_chain, chroma_db):
    # Run the full chain to get the response
    response = processing_chain.run({"user_input": user_input})

    # Check if the input was flagged as hate speech and store it if relevant
    if "and" in response:
        try:
            # Store user input if hate speech is detected and the distance is appropriate
            store_user_input(user_input, chroma_db)
        except Exception as e:
            print(f"Error storing user input after LLM check: {e}")

    return response


# Initialize model, prompt, ChromaDB, and process a sample input
model = initialize_model()
chroma_db = initialize_chromadb()
prompt_template = initialize_prompt_template()

# Create the processing chain
processing_chain = create_processing_chain(model, prompt_template, chroma_db)

# Example user input
user_input = "Sample text for hate speech detection"
response = process_user_input(user_input, processing_chain, chroma_db)
print(response)
