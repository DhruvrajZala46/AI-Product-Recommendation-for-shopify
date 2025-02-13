import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load product data from CSV
df = pd.read_csv("Shopify_Products - products_export_1 (1).csv")

# Ensure necessary columns exist
if "cleanDescription" not in df.columns or "Tags" not in df.columns:
    raise ValueError("CSV must contain 'cleanDescription' and 'Tags' columns.")

# Load the AI model (state-of-the-art NLP model)
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Handle missing values in 'cleanDescription' and 'Tags'
df["cleanDescription"] = df["cleanDescription"].fillna("")
df["Tags"] = df["Tags"].fillna("")

# Convert product descriptions + tags into AI embeddings
product_texts = (df["cleanDescription"] + " " + df["Tags"]).tolist()
product_embeddings = model.encode(product_texts, normalize_embeddings=True)

# Convert embeddings to NumPy format for FAISS
product_embeddings_np = np.array(product_embeddings).astype('float32')

# Create FAISS search index
d = product_embeddings_np.shape[1]
index = faiss.IndexFlatL2(d)
index.add(product_embeddings_np)

# Save FAISS index and updated CSV
faiss.write_index(index, "shopify_products.index")
df.to_csv("products_with_embeddings.csv", index=False)

print("✅ AI-powered product embeddings created successfully!")
