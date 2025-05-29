from pymilvus import (
    MilvusClient,
    CollectionSchema,
    DataType,
    FieldSchema,
    utility,
    Collection,
    connections
)

from langchain_ibm import WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the necessary environment variables
apikey = os.getenv("WATSONX_API_KEY")
url = os.getenv("WATSONX_URL")
project_id = os.getenv("WATSONX_PROJECT_ID")
URI = "http://localhost:19530"  # URI for Milvus
# URI = "./milvus.db"

# Initialize Milvus Collection
COLLECTION_NAME = "my_langgraph_app"

def connect_to_milvus():
    """Establish connection to Milvus server."""
    # Ensure a connection to the Milvus server is created
    connections.connect("default", uri=URI)
    print("Connected to Milvus server.")

def init_milvus_collection():
    """Creates a Milvus collection with the specified schema."""
    client = MilvusClient(URI)

    # Define schema for the collection
    idx = FieldSchema(
        name="id", dtype=DataType.INT64, is_primary=True
    )
    vector = FieldSchema(
        name="vector", dtype=DataType.FLOAT_VECTOR, dim=384  # Dimensionality of the vectors
    )
    doc_id = FieldSchema(
        name="document_id", dtype=DataType.INT64
    )
    text = FieldSchema(
        name="text", dtype=DataType.VARCHAR, max_length=65535
    )
#     src = FieldSchema(
#         name='src', dtype=DataType.VARCHAR, max_length=65536
#     )
    fields = [idx, vector, doc_id, text]
    schema = CollectionSchema(fields=fields)

    # Drop collection if exists and recreate
    if utility.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    # Create the collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        metric_type="L2",
        auto_id=True
    )
    print(f"Collection {COLLECTION_NAME} created.")

def init_embeddings():
    """Initialize Watsonx embeddings."""
    embeddings = WatsonxEmbeddings(
        model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
        url=url,
        apikey=apikey,
        project_id=project_id,
    )
    return embeddings

def load_documents_from_urls():
    """Load documents from a list of URLs."""
    urls = [
  "https://www.marketermilk.com/blog/best-ai-agent-platforms?utm_source=chatgpt.com",
  "https://www.forbes.com/sites/aytekintank/2025/02/27/the-top-150-ai-agents/?utm_source=chatgpt.com",
  "https://www.multimodal.dev/post/ai-agent-companies?utm_source=chatgpt.com",
  "https://fellow.app/blog/productivity/ai-agents-for-business/?utm_source=chatgpt.com",
  "https://keploy.io/blog/community/top-open-source-ai-agents?utm_source=chatgpt.com",
  "https://www.jotform.com/ai/agents/best-ai-agents/?utm_source=chatgpt.com",
  "https://aiagentsdirectory.com/?utm_source=chatgpt.com",
  "https://www.vox.com/technology/399512/ai-open-ai-operator-agents-paris-aiaction-summit?utm_source=chatgpt.com",
  "https://nymag.com/intelligencer/article/what-are-ai-agents-like-openai-operator-for.html?utm_campaign=feed-part&utm_medium=social_acct&utm_source=chatgpt.com",
  "https://www.businessinsider.com/ai-agents-jobs-board-ad-replacing-human-skills-2025-2?utm_source=chatgpt.com"
    ]


    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    return docs_list

def split_documents(docs_list):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits

def compute_embeddings(doc_splits, embeddings):
    """Compute embeddings for the split document chunks."""
    texts = [doc.page_content for doc in doc_splits]  # Extracting text from document chunks
    return embeddings.embed_documents(texts)  # Generating embeddings

def insert_data_into_milvus(batch):
    """Insert data (ID, vector, document ID, text) into Milvus."""
    client = MilvusClient(URI)
    client.insert(collection_name=COLLECTION_NAME, data=batch)
    print("Data inserted into Milvus.")

def create_vector_store_and_insert_data():
    """Main function to create a vector store in Milvus and insert document embeddings."""
    # Initialize Milvus collection if not created
    connect_to_milvus()

    init_milvus_collection()

    # Load documents
    docs_list = load_documents_from_urls()

    # Split documents into chunks
    doc_splits = split_documents(docs_list)

    # Initialize embeddings
    embeddings = init_embeddings()

    # Compute embeddings for document chunks
    embeddings_data = compute_embeddings(doc_splits, embeddings)

    # Prepare data to insert into Milvus
    batch = []
    for i, record in enumerate(doc_splits):
        batch.append({
            "id": i,
            "vector": embeddings_data[i],
            "document_id": i,
            "text": record.page_content
        })

    # Insert data into Milvus
    insert_data_into_milvus(batch)

    print("Vector store created and data inserted into Milvus.")

def create_index():
    """Create an index for the vector field in the collection."""
    client = MilvusClient(URI)

    index_params = client.prepare_index_params()

    # Add indexes
    index_params.add_index(
            field_name="vector",
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 1024}
      )

    # Create index on the "vector" field
    client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
    print(f"Index creation for collection {COLLECTION_NAME} started.")

def run_search(query):
    """Run a search query against the Milvus collection."""

    client = MilvusClient(URI)

    # Ensure the collection is loaded
    client.load_collection(COLLECTION_NAME)

    embeddings = init_embeddings()
    query_vector = embeddings.embed_documents([query])[0]

    search_params = {
        "metric_type": "L2"
    }

    # Search for similar vectors in the collection
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=5,
        output_fields=["id", "document_id", "text"],
        search_params=search_params
    )

    return results

def main():
    # Create vector store and insert data into Milvus
    create_vector_store_and_insert_data()
    create_index()

if __name__ == "__main__":
    main()
