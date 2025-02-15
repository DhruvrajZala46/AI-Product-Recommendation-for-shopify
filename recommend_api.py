from fastapi import FastAPI
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import uvicorn
import os  # Import os to handle dynamic port allocation

app = FastAPI()

# Load dataset & FAISS index
try:
    df = pd.read_csv("products_with_embeddings.csv")

    # Ensure no NaN values in dataframe
    df = df.astype(str).fillna("")

    # Load FAISS Index
    index = faiss.read_index("shopify_products.index")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    print("✅ Model & FAISS index loaded successfully.")

except Exception as e:
    print(f"❌ ERROR: {e}")
    df = None
    index = None
    model = None


@app.get("/recommend")
def recommend(query: str):
    try:
        if df is None or index is None or model is None:
            return {"error": "Server failed to load required files. Check logs."}

        print(f"🔍 Received Query: {query}")  # Log query

        # Convert query to AI embedding
        query_embedding = model.encode([query], normalize_embeddings=True)

        # Search FAISS for top 5 matches
        D, I = index.search(np.array(query_embedding).astype("float32"), k=5)
        print(f"🔢 FAISS Output Indices: {I[0]}")  # Log FAISS search results

        # Ensure valid indices
        valid_indices = [idx for idx in I[0] if 0 <= idx < len(df)]
        print(f"✅ Valid Indices: {valid_indices}")

        # Fetch product details
        recommendations = df.iloc[valid_indices][
            ["Title", "Description", "cleanDescription", "Product Category", "Type", "Tags", "Variant Price", "Image Src", "Image Position"]
        ].to_dict(orient="records")

        return {"recommendations": recommendations}

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {"error": str(e)}


# 🚀 Deployment entry point for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use Render's assigned port
    uvicorn.run(app, host="0.0.0.0", port=port)
