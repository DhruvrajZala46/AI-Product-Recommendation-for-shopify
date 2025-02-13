import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load product data from CSV
df = pd.read_csv("Shopify_Products - products_export_1 (1).csv")

# Ensure "cleanDescription" and "Tags" columns exist, fill missing values
df["cleanDescription"] = df["cleanDescription"].fillna("")
df["Tags"] = df["Tags"].fillna("")

# Load AI model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Convert product descriptions + tags into AI embeddings
product_texts = df["cleanDescription"] + " " + df["Tags"]
product_embeddings = model.encode(product_texts.tolist(), normalize_embeddings=True)

# Convert to NumPy format
product_embeddings_np = np.array(product_embeddings).astype("float32")

# Create FAISS index
d = product_embeddings_np.shape[1]
index = faiss.IndexFlatL2(d)
index.add(product_embeddings_np)

# Save FAISS index and updated CSV
faiss.write_index(index, "shopify_products.index")
df.to_csv("products_with_embeddings.csv", index=False)

print("✅ AI-powered product embeddings created successfully!")
