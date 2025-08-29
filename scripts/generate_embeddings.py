import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()
directory = "data/raw"
pages = []

# 1. List all files in the directory
for filename in os.listdir(directory):
    # 2. Check if the file is a PDF
    if filename.endswith(".pdf"):
        # 3. Create the full file path
        file_path = os.path.join(directory, filename)
        print(f"Loading {file_path}...")

        # 4. Load the PDF
        loader = PyPDFLoader(file_path)
        # 5. Use the standard synchronous load method
        doc = loader.load()
        # 6. Extend the pages list with the loaded document
        pages.extend(doc)

# Now `pages` contains all pages from all PDFs
pdf_count = sum(1 for f in os.listdir(directory) if f.endswith('.pdf'))
print(f"Loaded {len(pages)} pages from {pdf_count} PDF files.")

# Initialize embeddings model
embeddings_model = OpenAIEmbeddings()

# Create vector store with the embeddings model
vector_store = InMemoryVectorStore(embedding=embeddings_model)

# Add documents to the vector store
# The vector store will automatically handle embedding creation
vector_store.add_documents(pages)

print(f"Added {len(pages)} documents to vector store")

# Example: Query the vector store
query = "What is the main topic of these documents?"
results = vector_store.similarity_search(query, k=3)

print(f"\nTop {len(results)} results for query: '{query}'")
print("-" * 50)

for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"Source: {result.metadata.get('source', 'Unknown')}")
    print(f"Page: {result.metadata.get('page', 'N/A')}")
    print(f"Content: {result.page_content[:200]}...")  # First 200 chars
