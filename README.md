# RealityOS

RealityOS turns documents into actionable information. Upload a document, extract its text, identify important dates and amounts, and manage the resulting actions from one dashboard.

## Stack
- React + Vite
- FastAPI + SQLAlchemy
- PostgreSQL
- JWT authentication
- PyMuPDF for PDF text extraction

## Development
### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Set `VITE_API_URL=http://localhost:8000` for the frontend if needed.

The application intentionally contains no deployment infrastructure. Docker, Kubernetes, Terraform, CI/CD and production infrastructure are separate work for the DevOps stage.
