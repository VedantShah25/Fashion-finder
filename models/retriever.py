from utils.similarity import SimilaritySearch

class Retriever:
    def __init__(self, df):
        self.df = df
        self.search_engine = SimilaritySearch(df["Embedding"].tolist())

    def retrieve(self, query_embedding, top_k=5):
        _, indices = self.search_engine.search(query_embedding, top_k=top_k)
        return self.df.iloc[indices]

