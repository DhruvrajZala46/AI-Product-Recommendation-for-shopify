from fastapi import FastAPI
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()

# Load dataset & FAISS index
df = pd.read_csv("products_with_embeddings.csv")

# Ensure no NaN values in dataframe
df = df.astype(str).fillna("")

# Load FAISS Index
index = faiss.read_index("shopify_products.index")
model = SentenceTransformer("BAAI/bge-base-en-v1.5")


@app.get("/recommend")
def recommend(query: str):
    try:
        # Convert query to AI embedding
        query_embedding = model.encode([query], normalize_embeddings=True)

        # Search FAISS for top 5 matches
        D, I = index.search(np.array(query_embedding).astype("float32"), k=5)

        # Ensure valid indices (avoid out-of-bounds)
        valid_indices = [idx for idx in I[0] if idx >= 0 and idx < len(df)]

        # Fetch product details
        recommendations = df.iloc[valid_indices][
            ["Title", "Description", "cleanDescription", "Product Category", "Type", "Tags", "Variant Price", "Image Src", "Image Position"]
        ].to_dict(orient="records")

        return {"recommendations": recommendations}

    except Exception as e:
        return {"error": str(e)}


# 🚀 Deployment entry point for Render
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
