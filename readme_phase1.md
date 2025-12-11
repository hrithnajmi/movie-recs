# Movie & Book Recommendation Chatbot

A terminal-based chatbot that recommends movies and books using RAG (Retrieval-Augmented Generation) and multiple APIs.

## Features

- 🎬 Movie recommendations using OMDB API
- 📚 Book information using Open Library API
- 🔍 Web search using Brave Search API
- 🧠 RAG-powered personal knowledge base
- 🤖 LLM-powered intelligent responses

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt
```

### 2. Get API Keys

1. **Brave Search**: https://brave.com/search/api/
2. **OMDB**: http://www.omdbapi.com/apikey.aspx
3. **Hugging Face**: https://huggingface.co/settings/tokens

### 3. Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env and add your API keys
```

### 4. Add Your Documents

Place your movie/book documents in the `documents/` folder.

## Usage

```bash
python main.py
```

## Project Structure

```
movie-book-bot/
├── main.py           # Main chatbot
├── tools.py          # Tool definitions
├── rag.py            # RAG system
├── config.py         # Configuration
├── documents/        # Your documents
└── requirements.txt  # Dependencies
```

## Coming Soon

- [ ] Phase 1: Project setup ✅
- [ ] Phase 2: RAG system
- [ ] Phase 3: Tools implementation
- [ ] Phase 4: LLM integration
- [ ] Phase 5: Chat interface
- [ ] Phase 6: Testing & refinement

## License

MIT