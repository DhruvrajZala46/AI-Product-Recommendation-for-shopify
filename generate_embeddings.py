import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load product data from CSV
df = pd.read_csv("Shopify_Products - products_export_1 (1).csv")

# Ensure necessary columns exist
if "cleanDescription" not in df.columns or "Tags" not in df.columns:
    raise ValueError("CSV must contain 'cleanDescription' and 'Tags' columns.")

# Load a smaller and optimized AI model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Handle missing values and convert all data to string format
df["cleanDescription"] = df["cleanDescription"].fillna("").astype(str)
df["Tags"] = df["Tags"].fillna("").astype(str)

# Combine product descriptions and tags for embedding generation
product_texts = (df["cleanDescription"] + " " + df["Tags"]).tolist()

# Generate AI embeddings (convert text into numerical vectors)
product_embeddings = model.encode(product_texts, normalize_embeddings=True)

# Convert embeddings to NumPy array for FAISS
product_embeddings_np = np.array(product_embeddings).astype('float32')

# Define FAISS index with optimized memory usage
d = product_embeddings_np.shape[1]  # Embedding dimension
quantizer = faiss.IndexFlatL2(d)  # Quantizer for IVF index
nlist = 10  # Number of clusters for memory optimization
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)  # Create index

# Train and add embeddings to FAISS index
index.train(product_embeddings_np)
index.add(product_embeddings_np)

# Save FAISS index and updated CSV
faiss.write_index(index, "shopify_products.index")
df.to_csv("products_with_embeddings.csv", index=False)

print("✅ AI-powered product embeddings created successfully!")
