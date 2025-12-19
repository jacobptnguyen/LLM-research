import os
import datetime
import chromadb
from sentence_transformers import SentenceTransformer
from newsapi import NewsApiClient
from dotenv import load_dotenv
load_dotenv()

##here we are defining the chroma directory and the vector collection name
CHROMA_DIR = "./chroma_store"
VECTOR_COLLECTION = "startup_finance_rag"

##here we are loading the sentence transformer model and the chroma client
model = SentenceTransformer("all-MiniLM-L6-v2")
vector_client = chromadb.PersistentClient(path=CHROMA_DIR)
vector_collection = vector_client.get_collection(VECTOR_COLLECTION)

##here we are loading the news api client
try:
    news_client = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))
except KeyError as exc:
    raise EnvironmentError("Set NEWSAPI_KEY before calling news_lookup.") from exc

##here we are defining the semantic retriever function
def semantic_retriever(query: str, k: int = 2) -> list[str]:
    ##here we are encoding the query and getting the results from the vector collection
    embedding = model.encode(query).tolist()
    results = vector_collection.query(query_embeddings=[embedding], n_results=k)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return [
        f"Source: {meta.get('source', 'unknown')} | Evidence: {doc}"
        for doc, meta in zip(docs, metas)
    ]

##here we are defining the news lookup function
def news_lookup(query: str) -> list[str]:
    ##here we are getting the news articles from the news api
    data = news_client.get_everything(
        q=query,
        language="en",
        sort_by="publishedAt",
        from_param="2025-11-18",  # or compute last 7 days
        page_size=3,
    )
    ##here we are returning the news articles
    return [
        f"{art['title']} ({art['source']['name']})\n{art['description']}\n{art['url']}"
        for art in data.get("articles", [])
    ]

