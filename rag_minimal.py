import os, re
from typing import List, Dict, Tuple
import numpy as np
import tiktoken

try

# --- Read API key if you still want OpenAI as last-resort fallback ---
if os.path.exists("vst.openai.api.key"):
    with open("vst.openai.api.key", "r", encoding="utf-8") as f:
        key = f.read().strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key

# Avoid transformers pulling in TF when present
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# -----------------------------
# Config
# -----------------------------
MODEL_GENERATION = "gpt-4o-mini"     # used only for final answer
CHUNK_TOKENS    = 350
CHUNK_OVERLAP   = 60
TOP_K           = 6
TEMPERATURE     = 0.2

# Embedding backends preference (you can force one by editing this list)
EMBEDDING_BACKENDS = ["fastembed", "sentence_transformers", "openai"]
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"         # ~384-dim, fast, good quality
E5_MODEL        = "intfloat/e5-base-v2"            # if you keep sentence-transformers
OPENAI_EMBEDDER = "text-embedding-3-small"         # last resort (costs tokens)

# -----------------------------
# Example corpus
# -----------------------------
DOCS: List[Dict] = [
    {
        "id": "news_2025_policy_overview",
        "source": "Example Source A",
        "text": """
The Executive Yuan announced an updated national AI strategy in 2025 focusing on 
(1) safety and governance, (2) semiconductor-enabled AI infrastructure, 
(3) talent development and research, and (4) public-sector AI adoption. 
Measures include a national compute program, model evaluation sandboxes, 
SME digital transformation grants, and privacy-by-design standards.
"""
    },
    {
        "id": "press_release_safety",
        "source": "Example Source B",
        "text": """
The government introduced AI Safety Guidelines requiring risk assessment for 
high-impact models, disclosure of evaluation results, incident reporting within 72 hours, 
and red-teaming for critical sectors such as healthcare and finance.
"""
    },
    {
        "id": "program_funding",
        "source": "Example Source C",
        "text": """
A three-year funding program supports AI startups with cloud compute credits, 
access to public datasets via secure enclaves, and co-innovation projects with hospitals and universities. 
Universities receive grants to build courses on ML Ops and AI security.
"""
    },
    {
        "id": "public_sector_adoption",
        "source": "Example Source D",
        "text": """
Public agencies are encouraged to deploy AI assistants under strict procurement rules. 
Requirements include human-in-the-loop review, bias testing, and audit logs for decisions 
affecting citizen benefits.
"""
    },
]

# -----------------------------
# Utilities
# -----------------------------
def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def tokenize_chunks(text: str, max_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    ids = enc.encode(clean_text(text))
    chunks, start = [], 0
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        chunks.append(enc.decode(ids[start:end]))
        if end == len(ids): break
        start = end - overlap
    return chunks

# -----------------------------
# Embedding backends
# -----------------------------
class Embedder:
    def __init__(self):
        self.kind = None
        self.dim = None

        self._fe = None
        self._st = None
        self._openai = None

        for backend in EMBEDDING_BACKENDS:
            if backend == "fastembed":
                try:
                    from fastembed import TextEmbedding
                    self._fe = TextEmbedding(model_name=FASTEMBED_MODEL)
                    # probe dim by embedding one sentence
                    v = next(self._fe.embed(["probe sentence"]))
                    self.dim = len(v)
                    self.kind = "fastembed"
                    print(f"[embed] Using FastEmbed: {FASTEMBED_MODEL} (dim={self.dim})")
                    break
                except Exception as e:
                    print(f"[warn] FastEmbed unavailable: {type(e).__name__}: {e}")

            elif backend == "sentence_transformers":
                try:
                    from sentence_transformers import SentenceTransformer
                    self._st = SentenceTransformer(E5_MODEL)
                    self.dim = self._st.get_sentence_embedding_dimension()
                    self.kind = "sentence_transformers"
                    print(f"[embed] Using SentenceTransformers: {E5_MODEL} (dim={self.dim})")
                    break
                except Exception as e:
                    print(f"[warn] SentenceTransformers unavailable: {type(e).__name__}: {e}")

            elif backend == "openai":
                try:
                    from openai import OpenAI
                    self._openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                    # OpenAI dims for text-embedding-3-small = 1536
                    self.dim = 1536
                    self.kind = "openai"
                    print(f"[embed] Using OpenAI embeddings: {OPENAI_EMBEDDER} (dim={self.dim})")
                    break
                except Exception as e:
                    print(f"[warn] OpenAI embeddings unavailable: {type(e).__name__}: {e}")

        if self.kind is None:
            raise RuntimeError("No embedding backend available. "
                               "Install 'fastembed' (recommended) or provide OPENAI_API_KEY.")

    def embed_docs(self, texts: List[str]) -> np.ndarray:
        if self.kind == "fastembed":
            # FastEmbed expects raw passages; normalize for cosine
            vecs = np.array(list(self._fe.embed(texts)), dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
            return (vecs / norms).astype(np.float32)

        if self.kind == "sentence_transformers":
            # e5: use "passage: " for docs
            payload = ["passage: " + t for t in texts]
            vecs = self._st.encode(payload, normalize_embeddings=True)
            return vecs.astype(np.float32)

        if self.kind == "openai":
            resp = self._openai.embeddings.create(model=OPENAI_EMBEDDER, input=texts)
            arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
            return (arr / norms).astype(np.float32)

        raise RuntimeError("Unknown embedding backend")

    def embed_query(self, text: str) -> np.ndarray:
        if self.kind == "sentence_transformers":
            text = "query: " + text  # e5 query prefix
        if self.kind == "fastembed":
            # FastEmbed works fine with raw text for queries
            pass
        if self.kind == "openai":
            pass
        return self.embed_docs([text])[0]

# -----------------------------
# Vector index: try FAISS → NumPy fallback
# -----------------------------
USE_FAISS = True
try:
    import faiss  # type: ignore
except Exception as e:
    print(f"[info] FAISS unavailable, using NumPy index: {type(e).__name__}: {e}")
    USE_FAISS = False

class VectorStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.dim = embedder.dim
        self.meta: List[Dict] = []
        self._mat: np.ndarray = None
        self.index = faiss.IndexFlatIP(self.dim) if USE_FAISS else None

    def add_documents(self, docs: List[Dict]):
        chunks = []
        for d in docs:
            for ch in tokenize_chunks(d["text"]):
                chunks.append({"doc_id": d["id"], "source": d["source"], "chunk": ch})
        if not chunks:
            raise RuntimeError("No chunks to index.")

        vecs = self.embedder.embed_docs([c["chunk"] for c in chunks])
        self.meta.extend(chunks)

        if USE_FAISS:
            self.index.add(vecs)
        else:
            self._mat = vecs if self._mat is None else np.vstack([self._mat, vecs])

        print(f"Indexed {vecs.shape[0]} chunks (dim={self.dim}). Backend: "
              f"{'FAISS' if USE_FAISS else 'NumPy'}, Embedder: {self.embedder.kind}")

    def query(self, q: str, top_k: int = TOP_K) -> List[Tuple[float, Dict]]:
        qv = self.embedder.embed_query(q).reshape(1, -1)
        if USE_FAISS:
            D, I = self.index.search(qv.astype(np.float32), top_k)
            out = []
            for score, idx in zip(D[0], I[0]):
                if 0 <= int(idx) < len(self.meta):
                    out.append((float(score), self.meta[int(idx)]))
            return out
        else:
            sims = (self._mat @ qv.T).ravel()
            idx = np.argsort(-sims)[:top_k]
            return [(float(sims[i]), self.meta[i]) for i in idx]

# -----------------------------
# Prompt assembly
# -----------------------------
def build_prompt(question: str, contexts: List[Dict]) -> str:
    uniq, seen = [], set()
    for c in contexts:
        key = (c["doc_id"], c["chunk"][:120])
        if key not in seen:
            seen.add(key)
            uniq.append(c)

    sources_block = ""
    for i, c in enumerate(uniq, 1):
        sources_block += f"\n[S{i}] {c['source']} ({c['doc_id']})\n{c['chunk']}\n"

    return f"""You are a careful analyst.
Use ONLY the sources provided. If insufficient, say so briefly.
Cite by [S1], [S2], ... inline.

Question:
{question}

Sources:
{sources_block}

Answer (concise, factual, with inline citations):"""

# -----------------------------
# OpenAI generation (same as before)
# -----------------------------
def call_llm(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=MODEL_GENERATION,
        messages=[{"role":"user","content":prompt}],
        temperature=TEMPERATURE,
    )
    return resp.choices[0].message.content

# -----------------------------
# Demo
# -----------------------------
def main():
    question = "What are Taiwan’s latest AI policies in 2025?"
    embedder = Embedder()  # will pick FastEmbed first
    vs = VectorStore(embedder)
    vs.add_documents(DOCS)
    hits = vs.query(question, top_k=TOP_K)
    contexts = [h[1] for h in hits]
    prompt = build_prompt(question, contexts)
    print("\n--- PROMPT PREVIEW ---\n")
    print(prompt[:1200], "...\n")
    print("--- GENERATION ---\n")
    print(call_llm(prompt))

if __name__ == "__main__":
    main()
