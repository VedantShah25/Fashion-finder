from utils.similarity import SimilaritySearch

class Retriever:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.search_engine = SimilaritySearch(df["Embedding"].tolist())

    def retrieve(self, query_embedding, top_k=5):
        _, indices = self.search_engine.search(query_embedding, top_k=top_k * 3)
        results = self.df.iloc[indices].copy()
        results["similarity_score"] = _
        
        if "Image URL" in results.columns:
            results = results.drop_duplicates(subset=["Image URL"])
        else:
            results = results.drop_duplicates(subset=["Item Name"])
            
        return results.sort_values("similarity_score", ascending=True).head(top_k)

