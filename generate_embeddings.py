import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load only required columns from CSV (reduces memory usage)
df = pd.read_csv("Shopify_Products.csv", usecols=["cleanDescription", "Tags"])

# Fill missing values properly
df["cleanDescription"] = df["cleanDescription"].astype(str).fillna("")
df["Tags"] = df["Tags"].astype(str).fillna("")

# Load AI Model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Generate embeddings
product_texts = (df["cleanDescription"] + " " + df["Tags"]).tolist()
product_embeddings = model.encode(product_texts, normalize_embeddings=True)

# Convert embeddings to NumPy
product_embeddings_np = np.array(product_embeddings, dtype=np.float32)

# Create FAISS index
d = product_embeddings_np.shape[1]
index = faiss.IndexFlatL2(d)
index.add(product_embeddings_np)

# Save FAISS index
faiss.write_index(index, "shopify_products.index")

# Save updated CSV
df.to_csv("products_with_embeddings.csv", index=False)

print("✅ AI-powered product embeddings created successfully!")
