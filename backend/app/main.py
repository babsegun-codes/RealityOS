from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import json

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, create_engine, select, func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
import fitz

from .ai_engine import analyze_document

DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/realityos"
SECRET_KEY = "replace-with-your-own-development-secret"
ALGORITHM = "HS256"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    documents: Mapped[list["Document"]] = relationship(back_populates="owner", cascade="all, delete-orphan")
    actions: Mapped[list["Action"]] = relationship(back_populates="owner", cascade="all, delete-orphan")

class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    stored_path: Mapped[str] = mapped_column(String(500))
    extracted_text: Mapped[str] = mapped_column(Text, default="")
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    owner: Mapped[User] = relationship(back_populates="documents")
    actions: Mapped[list["Action"]] = relationship(back_populates="document", cascade="all, delete-orphan")

class Action(Base):
    __tablename__ = "actions"
    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    document_id: Mapped[Optional[int]] = mapped_column(ForeignKey("documents.id"), nullable=True)
    title: Mapped[str] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    due_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    priority: Mapped[str] = mapped_column(String(20), default="medium")
    completed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    owner: Mapped[User] = relationship(back_populates="actions")
    document: Mapped[Optional[Document]] = relationship(back_populates="actions")

class Intelligence(Base):
    __tablename__ = "intelligence"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), unique=True)
    analysis_json: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(100), default="gpt-4.1-mini")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

class RegisterIn(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class ActionIn(BaseModel):
    title: str
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    priority: str = "medium"
    document_id: Optional[int] = None

class ActionOut(ActionIn):
    id: int
    completed: bool
    model_config = {"from_attributes": True}

app = FastAPI(title="RealityOS API", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

def token_for(user: User) -> str:
    return jwt.encode({"sub": str(user.id), "exp": datetime.utcnow() + timedelta(days=1)}, SECRET_KEY, algorithm=ALGORITHM)

def current_user(token: str = Depends(oauth2_scheme), session: Session = Depends(db)) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def analyse_text(text: str):
    import re
    dates = re.findall(r"\b(?:\d{1,2}[/-]){2}\d{2,4}\b", text)
    amounts = re.findall(r"(?:₦|NGN|\$|USD)\s?[\d,]+(?:\.\d{2})?", text, re.I)
    return dates[:10], amounts[:10]

def save_ai_actions(analysis: dict, doc: Document, user: User, session: Session):
    obligations = analysis.get("obligations", []) or []
    created = []
    existing = {a.title for a in session.scalars(select(Action).where(Action.owner_id == user.id, Action.document_id == doc.id)).all()}
    for item in obligations:
        title = str(item.get("title", "")).strip()
        if not title or title in existing:
            continue
        due = None
        raw_date = item.get("due_date")
        if raw_date:
            try:
                due = datetime.fromisoformat(str(raw_date))
            except ValueError:
                due = None
        priority = str(item.get("priority", "medium")).lower()
        if priority not in {"low", "medium", "high", "critical"}:
            priority = "medium"
        action = Action(
            owner_id=user.id,
            document_id=doc.id,
            title=title[:255],
            description=item.get("description") or item.get("consequence"),
            due_date=due,
            priority=priority,
        )
        session.add(action)
        created.append(action)
        existing.add(title)
    session.commit()
    return created

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "realityos", "intelligence": "enabled"}

@app.post("/api/auth/register")
def register(data: RegisterIn, session: Session = Depends(db)):
    if session.scalar(select(User).where(User.email == data.email.lower())):
        raise HTTPException(400, "Email already registered")
    user = User(name=data.name, email=data.email.lower(), password_hash=pwd_context.hash(data.password))
    session.add(user)
    session.commit()
    session.refresh(user)
    return {"access_token": token_for(user), "user": {"id": user.id, "name": user.name, "email": user.email}}

@app.post("/api/auth/login")
def login(data: LoginIn, session: Session = Depends(db)):
    user = session.scalar(select(User).where(User.email == data.email.lower()))
    if not user or not pwd_context.verify(data.password, user.password_hash):
        raise HTTPException(401, "Invalid email or password")
    return {"access_token": token_for(user), "user": {"id": user.id, "name": user.name, "email": user.email}}

@app.get("/api/me")
def me(user: User = Depends(current_user)):
    return {"id": user.id, "name": user.name, "email": user.email}

@app.post("/api/documents")
def upload_document(file: UploadFile = File(...), user: User = Depends(current_user), session: Session = Depends(db)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported in the first release")
    path = UPLOAD_DIR / f"{user.id}_{datetime.utcnow().timestamp()}_{file.filename}"
    data = file.file.read()
    path.write_bytes(data)
    try:
        pdf = fitz.open(stream=data, filetype="pdf")
        text = "\n".join(page.get_text() for page in pdf)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read PDF: {exc}")
    dates, amounts = analyse_text(text)
    summary = f"Detected {len(dates)} date(s) and {len(amounts)} monetary amount(s)."
    doc = Document(owner_id=user.id, filename=file.filename, content_type=file.content_type or "application/pdf", stored_path=str(path), extracted_text=text, summary=summary)
    session.add(doc)
    session.commit()
    session.refresh(doc)
    return {"id": doc.id, "filename": doc.filename, "summary": summary, "dates": dates, "amounts": amounts, "text_preview": text[:1000]}

@app.get("/api/documents")
def documents(user: User = Depends(current_user), session: Session = Depends(db)):
    docs = session.scalars(select(Document).where(Document.owner_id == user.id).order_by(Document.created_at.desc())).all()
    return [{"id": d.id, "filename": d.filename, "summary": d.summary, "created_at": d.created_at, "ai_ready": bool(d.extracted_text)} for d in docs]

@app.get("/api/documents/{document_id}")
def document(document_id: int, user: User = Depends(current_user), session: Session = Depends(db)):
    doc = session.scalar(select(Document).where(Document.id == document_id, Document.owner_id == user.id))
    if not doc:
        raise HTTPException(404, "Document not found")
    dates, amounts = analyse_text(doc.extracted_text)
    intelligence = session.scalar(select(Intelligence).where(Intelligence.document_id == doc.id))
    analysis = json.loads(intelligence.analysis_json) if intelligence else None
    return {"id": doc.id, "filename": doc.filename, "summary": doc.summary, "text": doc.extracted_text, "dates": dates, "amounts": amounts, "analysis": analysis}

@app.post("/api/documents/{document_id}/analyze")
def analyze(document_id: int, user: User = Depends(current_user), session: Session = Depends(db)):
    doc = session.scalar(select(Document).where(Document.id == document_id, Document.owner_id == user.id))
    if not doc:
        raise HTTPException(404, "Document not found")
    if not doc.extracted_text.strip():
        raise HTTPException(400, "No text was extracted from this document")
    try:
        analysis = analyze_document(doc.extracted_text, datetime.utcnow().date().isoformat())
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    existing = session.scalar(select(Intelligence).where(Intelligence.document_id == doc.id))
    model = "gpt-4.1-mini"
    if existing:
        existing.analysis_json = json.dumps(analysis, ensure_ascii=False)
        existing.model = model
    else:
        session.add(Intelligence(document_id=doc.id, analysis_json=json.dumps(analysis, ensure_ascii=False), model=model))
    save_ai_actions(analysis, doc, user, session)
    return {"document_id": doc.id, "analysis": analysis}

@app.get("/api/documents/{document_id}/intelligence")
def intelligence(document_id: int, user: User = Depends(current_user), session: Session = Depends(db)):
    doc = session.scalar(select(Document).where(Document.id == document_id, Document.owner_id == user.id))
    if not doc:
        raise HTTPException(404, "Document not found")
    record = session.scalar(select(Intelligence).where(Intelligence.document_id == doc.id))
    if not record:
        raise HTTPException(404, "Document has not been analyzed yet")
    return {"document_id": doc.id, "model": record.model, "analysis": json.loads(record.analysis_json), "created_at": record.created_at}

@app.get("/api/actions", response_model=list[ActionOut])
def actions(user: User = Depends(current_user), session: Session = Depends(db)):
    return session.scalars(select(Action).where(Action.owner_id == user.id).order_by(Action.completed, Action.due_date)).all()

@app.post("/api/actions", response_model=ActionOut)
def create_action(data: ActionIn, user: User = Depends(current_user), session: Session = Depends(db)):
    action = Action(owner_id=user.id, **data.model_dump())
    session.add(action)
    session.commit()
    session.refresh(action)
    return action

@app.patch("/api/actions/{action_id}/complete")
def complete_action(action_id: int, user: User = Depends(current_user), session: Session = Depends(db)):
    action = session.scalar(select(Action).where(Action.id == action_id, Action.owner_id == user.id))
    if not action:
        raise HTTPException(404, "Action not found")
    action.completed = True
    session.commit()
    return {"success": True}

@app.get("/api/dashboard")
def dashboard(user: User = Depends(current_user), session: Session = Depends(db)):
    docs = session.scalar(select(func.count(Document.id)).where(Document.owner_id == user.id))
    actions = session.scalars(select(Action).where(Action.owner_id == user.id)).all()
    return {
        "documents": docs or 0,
        "open_actions": sum(not a.completed for a in actions),
        "overdue": sum(not a.completed and a.due_date and a.due_date < datetime.utcnow() for a in actions),
        "completed": sum(a.completed for a in actions),
    }
