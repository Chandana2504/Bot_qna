import pandas as pd
import numpy as np
import faiss
import re 
import os
from sentence_transformers import SentenceTransformer

class FinancialChatbot:
    def __init__(self, trades_csv, holdings_csv):
        """
        INITIALIZATION: This runs once when you start the bot.
        It loads your data, sets up the AI model, and builds the search index.
        """
        # 1. Load your CSV files into Pandas DataFrames (tables)
        self.df_trades = pd.read_csv(trades_csv)
        self.df_holdings = pd.read_csv(holdings_csv)
        
        # 2. Load the AI Model (converts text into 'embeddings' or math vectors)
        print("Initializing AI Model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # 3. Process the tables into long strings of text that the AI can read
        print("Building Search Index...")
        self.all_texts = self._prepare_search_data()
        
        # 4. Create the FAISS Index (a high-speed search engine for AI vectors)
        self.index = self._create_faiss_index()
        
        # 5. Build the Entity Map (a list of all IDs and Tickers for 100% accuracy)
        self.entities = self._extract_entities()

    def _prepare_search_data(self):
        """
        DATA CLEANING: Converts every row in your CSV into a readable sentence.
        Example: [Trade] id: 3496826 | Ticker: META | Price: 108.0 ...
        """
        def row_to_text(row, label):
            # Creates a list of "ColumnName: Value" pairs, ignoring empty cells (NaN)
            items = [f"{col}: {val}" for col, val in row.items() if pd.notnull(val) and str(val).lower() != 'nan']
            return f"[{label}] " + " | ".join(items)
        
        # Convert all rows from both files into a single list of text strings
        t_texts = self.df_trades.apply(lambda r: row_to_text(r, "Trade"), axis=1).tolist()
        h_texts = self.df_holdings.apply(lambda r: row_to_text(r, "Holding"), axis=1).tolist()
        return t_texts + h_texts

    def _create_faiss_index(self):
        """
        AI VECTOR SEARCH SETUP: Turns the text into math so the bot can understand 'meaning'.
        """
        embeddings = self.model.encode(self.all_texts) # Convert text to numbers
        faiss.normalize_L2(embeddings)                # Standardize values for better matching
        index = faiss.IndexFlatIP(embeddings.shape[1]) # Create the search index
        index.add(np.array(embeddings).astype('float32'))
        return index

    def _extract_entities(self):
        """
        ACCURACY LAYER: Creates a 'dictionary' of Tickers and IDs.
        This prevents the bot from guessing when a specific ID is mentioned.
        """
        entities = {}
        for text in self.all_texts:
            # Match 7-digit Trade IDs (Unique)
            ids = re.findall(r'id: (\d+)', text)
            for i in ids: 
                entities[i] = text
            
            # Match Tickers (SPOT, IBM, etc.) 
            # We store these as a list because one Ticker can have many trades.
            tickers = re.findall(r'Ticker: ([\w\-]+)', text)
            for t in tickers:
                t_lower = t.lower()
                if t_lower not in entities:
                    entities[t_lower] = [] 
                if isinstance(entities[t_lower], list):
                    entities[t_lower].append(text)
        return entities

    def ask(self, question):
        """
        THE BRAIN: Decides which logic to use based on your question.
        """
        q_clean = question.lower()

        # --- STRATEGY 1: EXACT ID or TICKER MATCH ---
        # Highest priority. If you say "ID 1234567" or "SPOT", it does an exact lookup.
        
        # 1a. Look for 7-digit IDs
        found_ids = re.findall(r'\b\d{7}\b', question)
        if found_ids and found_ids[0] in self.entities:
            return f"(Exact ID Match found for {found_ids[0]}):\n{self.entities[found_ids[0]]}"

        # 1b. Look for Ticker names as standalone words (\b avoids matching 'al' in 'total')
        for ticker, results in self.entities.items():
            if re.search(rf'\b{re.escape(ticker)}\b', q_clean) and not ticker.isdigit():
                if isinstance(results, list):
                    all_matches = "\n\n---\n\n".join(results)
                    return f"(Found {len(results)} matches for ticker {ticker.upper()}):\n\n{all_matches}"
                return f"(Ticker Match):\n{results}"

        # --- STRATEGY 2: ANALYTICS (P&L MATH) ---
        # If the bot hears "best", "worst", or "total", it uses Pandas to do calculation.
        if any(word in q_clean for word in ["best", "worst", "performed", "total", "sum"]):
            df = self.df_holdings
            # Check if user is asking about P&L performance
            if "pl_ytd" in str(df.columns).lower() or "pl" in q_clean:
                # Group data by Portfolio and add up the P&L column
                perf = df.groupby('PortfolioName')['PL_YTD'].sum().sort_values(ascending=False)
                
                if "best" in q_clean or "top" in q_clean:
                    return f"The best performing fund is {perf.index[0]} with a total PL_YTD of {perf.iloc[0]:,.2f}."
                if "worst" in q_clean:
                    return f"The worst performing fund is {perf.index[-1]} with a total PL_YTD of {perf.iloc[-1]:,.2f}"
                if "total" in q_clean:
                    return f"The total P&L across all holdings is {df['PL_YTD'].sum():,.2f}"

        # --- STRATEGY 3: SEMANTIC SEARCH (AI FUZZY MATCH) ---
        # If the top two strategies fail, use AI to find the most 'similar' sounding row.
        query_vec = self.model.encode([question])
        faiss.normalize_L2(query_vec)
        D, I = self.index.search(np.array(query_vec).astype('float32'), k=1)
        
        # Only return the answer if the AI is reasonably confident (> 0.35 score)
        if D[0][0] > 0.35:
            return self.all_texts[I[0][0]]
        
        return "I couldn't find a specific answer. Try asking for a Trade ID, a Ticker, or 'best performing fund'."

# --- EXECUTION ---
# Change these paths to your local folder location
TRADES_FILE = r"C:\Users\cn200\Downloads\trades.csv"
HOLDINGS_FILE = r"C:\Users\cn200\Downloads\holdings.csv"

if __name__ == "__main__":
    # Initialize the bot
    bot = FinancialChatbot(TRADES_FILE, HOLDINGS_FILE)
    print("\nBot is ready! Type 'exit' to stop.")
    
    # Start the conversation loop
    while True:
        user_input = input("\nAsk a question: ")
        if user_input.lower() == 'exit':
            break
        print("\nAnswer:", bot.ask(user_input))
        
        
      
