Financial Hybrid-Intelligence Chatbot
This project is a specialized AI agent designed to bridge the gap between unstructured natural language and structured financial data. While standard AI models often struggle with exact numbers and math, this bot uses a "Hybrid Search" architecture to ensure financial accuracy.

🧠 How It Works: The Three-Layer Engine
The bot doesn't just "guess" the answer. It routes every question through three distinct layers:

The Precision Layer (Regex & Entity Mapping): * Scans for specific 7-digit Trade IDs and Tickers.

Uses word-boundary detection to prevent common errors (e.g., distinguishing the ticker "AL" from the word "Total").

The Analytics Layer (Pandas Engine): * Recognizes keywords like "best," "worst," "sum," or "P&L."

Instead of searching for a row, it triggers a real-time calculation across the entire holdings.csv to rank portfolio performance.

The Semantic Layer (AI Vector Search): * Uses FAISS (Facebook AI Similarity Search) and the Sentence-Transformers model.

Finds rows based on meaning, allowing you to ask descriptive questions like "Who was the custodian for the Meta deal?"

🛠️ Technical Stack
Engine: Python 3.x

Vector Database: FAISS (Facebook AI Similarity Search)

NLP Model: all-MiniLM-L6-v2 (Sentence-Transformers)

Data Handling: Pandas & NumPy

Pattern Matching: Regular Expressions (Regex)

🚀 Key Features
✅ Ticker Multi-Match: If a ticker (like SPOT) has multiple trades or holdings, the bot returns a comprehensive history instead of just the first result.

✅ Contextual Awareness: Distinguishes between "Trades" (transactional events) and "Holdings" (current portfolio snapshots).

✅ Confidence Thresholds: Includes a safety mechanism that tells the user "I couldn't find a specific answer" if the AI confidence score is too low, preventing "hallucinations."

✅ Formatting: Outputs results in a clean, readable pipe-separated format for easy review.

📈 Sample Queries to Try
Exact ID: "What happened in trade 3496826?"

Performance: "Which fund had the best PL_YTD?"

Multi-record: "List all records for ticker IBM."

Fuzzy Search: "Show me equity trades handled by JP Morgan."

📂 File Structure
chatbot.py: The main logic containing the FinancialChatbot class.

trades.csv: Historical transaction data.

holdings.csv: Portfolio snapshot and P&L data.

