import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import asyncio

# Load environment variables
load_dotenv()

# Set environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_ENV = os.getenv("PINECONE_ENV")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Upload directory
UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)

# Create index if it doesn't exist
existing_indexes = [i["name"] for i in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="dotproduct",
        spec=spec,
    )
    # Wait until index is ready
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

# Connect to index
index = pc.Index(PINECONE_INDEX_NAME)

# Async function to load documents into vector store
async def load_vectorstore(uploaded_files, role: str, doc_id: str):
    embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())

        loader = PyPDFLoader(str(save_path))
        documents = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        # Prepare data for Pinecone
        texts = [chunk.page_content for chunk in chunks]
        ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": file.filename,
                "doc_id": doc_id,
                "role": role,
                "page": chunk.metadata.get("page", 0)
            }
            for i, chunk in enumerate(chunks)
        ]

        print(f"[INFO] Embedding {len(texts)} chunks from '{file.filename}'...")

        # Generate embeddings
        embeddings = await asyncio.to_thread(embed_model.embed_documents, texts)

        print(f"[INFO] Uploading {len(embeddings)} vectors to Pinecone...")
        with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
            index.upsert(vectors=list(zip(ids, embeddings, metadatas)))
            progress.update(len(embeddings))

        print(f"[DONE] Upload complete for: {file.filename}")
