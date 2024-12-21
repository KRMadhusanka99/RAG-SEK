## Setup
1. Copy `.env.example` to `.env`
2. Add your OpenAI API key to `.env`

your_project/
│
├── assets/
│   └── logo.png
│
├── documents/
│   ├── preloaded/
│   └── uploaded/
│
├── src/
│   ├── __init__.py
│   ├── document_store.py
│   ├── rag_system.py
│   └── app.py
│
└── requirements.txt

## Document Setup
1. Preloaded documents:
   - Place default PDFs in `documents/preloaded/`
   - These files are tracked in git and serve as initial knowledge base
2. User uploads:
   - Go to `documents/uploaded/`
   - Not tracked in git for privacy