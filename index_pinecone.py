from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import uuid

# ✅ Load a 1024-dim model (must match your Pinecone index)
model = SentenceTransformer("sentence-transformers/all-roberta-large-v1")

# ✅ Initialize Pinecone
pc = Pinecone(api_key="USEYOURKEY")
index = pc.Index("USEYOURINDEXNAME")  # Use your actual index name

# ✅ Example data
texts = [
    ("This is a drama example", "drama"),
    ("A high-action thriller scene", "action"),
    ("Romantic movie with emotional plot", "drama"),
    ("Explosions and car chases", "action")
]

# ✅ Create embeddings and structure for upsert
vectors = []
for i, (text, genre) in enumerate(texts):
    embedding = model.encode(text).tolist()
    vectors.append({
        "id": f"vec_{i}_{uuid.uuid4().hex[:8]}",
        "values": embedding,
        "metadata": {"genre": genre}
    })

# ✅ Upsert into a namespace
index.upsert(
    vectors=vectors,
    namespace="ns1"
)

# ✅ Query example with metadata filter
query_vector = model.encode("Intense action movie").tolist()

response = index.query(
    namespace="ns1",
    vector=query_vector,
    top_k=2,
    include_values=True,
    include_metadata=True,
    filter={"genre": {"$eq": "action"}}
)

print(response)
