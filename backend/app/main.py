from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
import fitz

DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/realityos"
SECRET_KEY = "replace-with-your-own-development-secret"
ALGORITHM = "HS256"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

class Base(DeclarativeBase): pass

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

app = FastAPI(title="RealityOS API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def db():
    session = SessionLocal()
    try: yield session
    finally: session.close()

def token_for(user: User) -> str:
    return jwt.encode({"sub": str(user.id), "exp": datetime.utcnow() + timedelta(days=1)}, SECRET_KEY, algorithm=ALGORITHM)

def current_user(token: str = Depends(oauth2_scheme), session: Session = Depends(db)) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    user = session.get(User, user_id)
    if not user: raise HTTPException(status_code=401, detail="User not found")
    return user

def analyse_text(text: str):
    import re
    dates = re.findall(r"\b(?:\d{1,2}[/-]){2}\d{2,4}\b", text)
    amounts = re.findall(r"(?:₦|NGN|\$|USD)\s?[\d,]+(?:\.\d{2})?", text, re.I)
    return dates[:10], amounts[:10]

@app.get("/api/health")
def health(): return {"status": "ok", "service": "realityos"}

@app.post("/api/auth/register")
def register(data: RegisterIn, session: Session = Depends(db)):
    if session.scalar(select(User).where(User.email == data.email.lower())):
        raise HTTPException(400, "Email already registered")
    user = User(name=data.name, email=data.email.lower(), password_hash=pwd_context.hash(data.password))
    session.add(user); session.commit(); session.refresh(user)
    return {"access_token": token_for(user), "user": {"id": user.id, "name": user.name, "email": user.email}}

@app.post("/api/auth/login")
def login(data: LoginIn, session: Session = Depends(db)):
    user = session.scalar(select(User).where(User.email == data.email.lower()))
    if not user or not pwd_context.verify(data.password, user.password_hash): raise HTTPException(401, "Invalid email or password")
    return {"access_token": token_for(user), "user": {"id": user.id, "name": user.name, "email": user.email}}

@app.get("/api/me")
def me(user: User = Depends(current_user)): return {"id": user.id, "name": user.name, "email": user.email}

@app.post("/api/documents")
def upload_document(file: UploadFile = File(...), user: User = Depends(current_user), session: Session = Depends(db)):
    if not file.filename.lower().endswith(".pdf"): raise HTTPException(400, "Only PDF files are supported in the first release")
    path = UPLOAD_DIR / f"{user.id}_{datetime.utcnow().timestamp()}_{file.filename}"
    data = file.file.read(); path.write_bytes(data)
    try:
        pdf = fitz.open(stream=data, filetype="pdf")
        text = "\n".join(page.get_text() for page in pdf)
    except Exception as exc:
        path.unlink(missing_ok=True); raise HTTPException(400, f"Could not read PDF: {exc}")
    dates, amounts = analyse_text(text)
    summary = f"Detected {len(dates)} date(s) and {len(amounts)} monetary amount(s)."
    doc = Document(owner_id=user.id, filename=file.filename, content_type=file.content_type or "application/pdf", stored_path=str(path), extracted_text=text, summary=summary)
    session.add(doc); session.commit(); session.refresh(doc)
    return {"id": doc.id, "filename": doc.filename, "summary": summary, "dates": dates, "amounts": amounts, "text_preview": text[:1000]}

@app.get("/api/documents")
def documents(user: User = Depends(current_user), session: Session = Depends(db)):
    docs = session.scalars(select(Document).where(Document.owner_id == user.id).order_by(Document.created_at.desc())).all()
    return [{"id": d.id, "filename": d.filename, "summary": d.summary, "created_at": d.created_at} for d in docs]

@app.get("/api/documents/{document_id}")
def document(document_id: int, user: User = Depends(current_user), session: Session = Depends(db)):
    doc = session.scalar(select(Document).where(Document.id == document_id, Document.owner_id == user.id))
    if not doc: raise HTTPException(404, "Document not found")
    dates, amounts = analyse_text(doc.extracted_text)
    return {"id": doc.id, "filename": doc.filename, "summary": doc.summary, "text": doc.extracted_text, "dates": dates, "amounts": amounts}

@app.get("/api/actions", response_model=list[ActionOut])
def actions(user: User = Depends(current_user), session: Session = Depends(db)):
    return session.scalars(select(Action).where(Action.owner_id == user.id).order_by(Action.completed, Action.due_date)).all()

@app.post("/api/actions", response_model=ActionOut)
def create_action(data: ActionIn, user: User = Depends(current_user), session: Session = Depends(db)):
    action = Action(owner_id=user.id, **data.model_dump())
    session.add(action); session.commit(); session.refresh(action); return action

@app.patch("/api/actions/{action_id}/complete")
def complete_action(action_id: int, user: User = Depends(current_user), session: Session = Depends(db)):
    action = session.scalar(select(Action).where(Action.id == action_id, Action.owner_id == user.id))
    if not action: raise HTTPException(404, "Action not found")
    action.completed = True; session.commit(); return {"success": True}

@app.get("/api/dashboard")
def dashboard(user: User = Depends(current_user), session: Session = Depends(db)):
    docs = session.scalar(select(__import__('sqlalchemy').func.count(Document.id)).where(Document.owner_id == user.id))
    actions = session.scalars(select(Action).where(Action.owner_id == user.id)).all()
    return {"documents": docs or 0, "open_actions": sum(not a.completed for a in actions), "overdue": sum(not a.completed and a.due_date and a.due_date < datetime.utcnow() for a in actions), "completed": sum(a.completed for a in actions)}
