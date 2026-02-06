# Entity Manager Agent System

A multi-agent RAG (Retrieval-Augmented Generation) system for managing and discovering entities (movies, books, etc.).

It uses a team of specialized AI agents to fetch new content, manage a local vector database, and provide personalized recommendations based on what you own.

## 🤖 The Agents

| Agent | Role | Capabilities |
|-------|------|--------------|
| **Dispatcher** | Orchestrator | Intelligently routes user queries to the right agent(s). Manages multi-step workflows. |
| **Scout** | Finder | Fetches NEW content from the internet (TMDB API). Finds movies, people, and details. |
| **Librarian** | Manager | Manages your personal collection. Adds, lists, deletes, and checks for existence of items. |
| **MovieCritic** | Recommender | RAG expert. Searches your collection semantically to find items matching moods/themes. |

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- TMDB API Key (for fetching new data)
- Groq API Key (or other LLM provider)

### Installation

1. **Clone the repo**
   ```bash
   git clone <repo-url>
   cd movie_helper
   ```

2. **Set up environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure keys**
   Create a `.env` file:
   ```env
   TMDB_API_KEY=your_tmdb_key
   GROQ_API_KEY=your_groq_key
   LLM_MODEL=openai/gpt-oss-20b  # or other compatible model
   ```

## 💡 Usage

Run the main script to interact with the system.

### Interactive Chat Mode (Recommended)
Start a continuous conversation session:

```bash
python main.py --chat
```

Once in chat mode, you can type naturally:
- "Fetch Inception"
- "Add it to my collection"
- "Suggest similar movies"

### Conversational Mode (One-Shot)
Execute a single command via the Dispatcher:

```bash
# Add a movie
python main.py "Add Inception to my collection"

# Check your collection
python main.py "Do I have The Dark Knight?"

# Ask for recommendations (RAG)
python main.py "Find me a dark sci-fi movie from my list"

# Manage collection
python main.py "Remove The Matrix"
```

### Verbose Mode
See what the agents are thinking:
```bash
python main.py -v "Add Interstellar"
```

### Direct Agent Access
Bypass the dispatcher to test specific agents:
```bash
python main.py --agent scout "Fetch Christopher Nolan movies"
python main.py --agent check_entity "Do I have Inception?"  # uses Librarian
```

## 🏗️ Architecture

- **Vector Store:** ChromaDB (stores entities with embeddings)
- **Framework:** LangChain (LLM orchestration)
- **LLM:** Groq (fast inference)

See [AGENTS.md](AGENTS.md) for detailed architecture documentation.
