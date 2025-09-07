import ollama

class Generator:
    def __init__(self, model="llama3"):
        self.model = model

    def generate(self, query, retrieved_items):
        context = "\n".join(
            [
                f"- {row['Item Name']} | Price: {row['Price']} | Link: {row['Link']}"
                for _, row in retrieved_items.iterrows()
            ]
        )
        prompt = f"""
        User query: {query}
        Retrieved similar products:
        {context}

        You are an expert fashion assistant.
        Based on the above, generate a helpful response for the user suggesting about the trending latest fashion styles.
        """
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]

