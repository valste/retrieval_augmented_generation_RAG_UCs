""" 
description: 
    A simple end-to-end example of Retrieval-Augmented Generation (RAG)
    for querying Taiwan AI policy using web scraping, embedding, and LLM.
    
step by step: 
    1. Collect candidate webpages (the “corpus”).
    2. Chunk and embed them.
    3. Index with FAISS.
    4. Retrieve top-k chunks for a given question.
    5. (Optional) Rerank them.
    6. Augment the prompt with those chunks and Generate the answer (with source links).
    
It will:
    * Crawl some Taiwan gov websites.
    * Build a vector index.
    * Retrieve relevant text chunks.
    * Send them + your question to GPT_MODEL.
    * Print an answer with citations.

notes:
    Token costs: GPT-4o mini is cheaper and faster—use it for most RAG use cases.
    Rate limits: Plus accounts usually have moderate API rate limits (fine for prototypes).
    Storage: The embeddings (via sentence-transformers) are local and don’t cost anything. You only pay for the final GPT-4o API calls.
    
"""

# TensorFlow being imported by transformers (via sentence_transformers) on Windows + Python 3.12. 
# You don’t need TF at all for RAG text embeddings, so fix it in one of these two ways

import os, re, requests
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup
from openai import OpenAI

import numpy as np
import faiss
from fastembed import TextEmbedding
#from sentence_transformers import SentenceTransformer
import tiktoken

# setting up the OpenAI API key environment variable to be able to access the OpenAI API
from openAI_apiKey import set_apiKey_env
set_apiKey_env()

###############################################################################
# 0) CONFIG
###############################################################################

ENCODER_NAME = "BAAI/bge-small-en-v1.5"   # Hugging Face model name for FastEmbed.
TOP_K = 8                              # of chunks to retrieve
CHUNK_TOKENS = 350
CHUNK_OVERLAP = 60
GPT_MODEL = "gpt-4o-mini"              # cheaper, faster, strong

# Put a few seed URLs about Taiwan AI policy (example). In production, collect more.
SEED_URLS = [
    # (Feel free to replace these with the sites you trust/need)
    "https://www.ndc.gov.tw/" ,        # Taiwan National Development Council
    "https://www.most.gov.tw/" ,       # (Now NSTC) National Science & Tech Council
    "https://www.ey.gov.tw/" ,         # Executive Yuan
    "https://english.ey.gov.tw/" ,
]


###############################################################################
# UTILS
###############################################################################
def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s

def fetch_url(url: str, timeout=20) -> str:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"[warn] fetch failed: {url} -> {e}")
        return ""

def extract_links(html: str, base: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        href = a["href"]
        if href.startswith("http"):
            links.append(href)
        elif href.startswith("/"):
            # naive join
            from urllib.parse import urljoin
            links.append(urljoin(base, href))
    # lightweight filtering (keep same domains, English pages, docs/news pages, etc.)
    keep = []
    for u in links:
        if any(root in u for root in ["ndc.gov.tw","most.gov.tw","ey.gov.tw"]):
            if any(x in u.lower() for x in ["ai","policy","plan","strategy","news","press","program","industry"]):
                keep.append(u)
    return list(dict.fromkeys(keep))  # de-dup

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # drop nav/scripts
    for bad in soup(["script","style","nav","header","footer"]):
        bad.decompose()
    text = clean_text(soup.get_text(" "))
    return text

def tokenize_len(text: str, model="gpt-4o-mini"):
    # Use tiktoken heuristics just to size chunks sanely; model name doesn't have to exist locally.
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def chunk_text(text: str, max_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    ids = enc.encode(text)
    chunks = []
    start = 0
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        chunk = enc.decode(ids[start:end])
        chunks.append(chunk)
        if end == len(ids): break
        start = end - overlap
    return chunks

###############################################################################
# 1) CORPUS COLLECTION (tiny crawler)
###############################################################################
def build_corpus(seed_urls: List[str], limit_pages=20) -> List[Dict]:
    seen = set()
    pages = []

    frontier = list(seed_urls)
    while frontier and len(pages) < limit_pages:
        url = frontier.pop(0)
        if url in seen: 
            continue
        seen.add(url)

        html = fetch_url(url)
        if not html: 
            continue

        text = html_to_text(html)
        if len(text) < 500:  # skip too-short pages
            continue

        pages.append({"url": url, "text": text})
        # discover more pages lightly
        for nxt in extract_links(html, url):
            if nxt not in seen and len(frontier) < limit_pages*3:
                frontier.append(nxt)

    print(f"[info] collected {len(pages)} pages")
    return pages

###############################################################################
# 2) CHUNK + 3) EMBED + 4) INDEX
###############################################################################
class VectorStore:
    """
    A simple vector store using FastEmbed (BGE) for embeddings
    and FAISS for similarity search (cosine similarity via inner product).
    """

    def __init__(self, encoder_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize the vector store.
        
        Args:
            encoder_name: Hugging Face model name for FastEmbed.
                          Default is "BAAI/bge-small-en-v1.5" (fast + good quality).
        """
        # Load FastEmbed model
        self.model = TextEmbedding(model_name=encoder_name)
        
        # Probe one embedding to get dimensionality
        probe = next(self.model.embed(["probe sentence"]))
        self.dim = len(probe)

        # Metadata store: maps internal ids to {url, chunk}
        self.id2meta: Dict[int, Dict] = {}
        self.next_id = 0

        # FAISS index (inner product on normalized vectors ≈ cosine)
        self.index = faiss.IndexFlatIP(self.dim)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize vectors to unit length (L2 norm = 1).
        Required so inner product in FAISS equals cosine similarity.
        
        Args:
            x: 2D numpy array of vectors.
        
        Returns:
            Normalized 2D numpy array.
        """
        x = x.astype("float32")
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        return x / norms

    def add_documents(self, docs: List[Dict]):
        """
        Embed and index text chunks from documents.
        
        Args:
            docs: List of dicts, each with keys {"url":..., "text":...}.
                  Each document's text will be chunked, embedded, and added to FAISS.
        """
        texts, metas = [], []
        for d in docs:
            # Break document text into overlapping chunks
            for chunk in chunk_text(d["text"]):
                texts.append(chunk)
                metas.append({"url": d["url"], "chunk": chunk})

        if not texts:
            raise RuntimeError("No chunks to index.")

        # Embed chunks using FastEmbed and normalize
        vecs = np.array(list(self.model.embed(texts)), dtype="float32")
        vecs = self._normalize(vecs)

        # Store metadata and assign IDs
        for m in metas:
            self.id2meta[self.next_id] = m
            self.next_id += 1

        # Add vectors to FAISS index
        self.index.add(vecs)

        print(f"[info] indexed {vecs.shape[0]} chunks, dim={self.dim}, backend=FAISS")

    def _embed_query(self, q: str) -> np.ndarray:
        """
        Embed and normalize a query string.
        
        Args:
            q: Query string.
        
        Returns:
            Normalized 2D numpy array (1 x dim).
        """
        v = np.array(list(self.model.embed([q]))[0], dtype="float32").reshape(1, -1)
        return self._normalize(v)

    def query(self, q: str, top_k=5) -> List[Tuple[float, Dict]]:
        """
        Retrieve top-k most similar chunks for a query.
        
        Args:
            q: Query string.
            top_k: Number of results to return.
        
        Returns:
            List of (similarity_score, metadata_dict) tuples.
            metadata_dict contains {"url":..., "chunk":...}.
        """
        # Embed query
        qv = self._embed_query(q)

        # Search FAISS index
        D, I = self.index.search(qv, top_k)

        # Collect results with metadata
        results = []
        for score, idx in zip(D[0], I[0]):
            meta = self.id2meta.get(int(idx))
            if meta:
                results.append((float(score), meta))
        return results

###############################################################################
# 5) (Optional) RERANK (cross-encoder) – you can skip this initially
###############################################################################
# from sentence_transformers import CrossEncoder
# RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# def rerank(query, hits):
#     pairs = [(query, h[1]["chunk"]) for h in hits]
#     scores = RERANKER.predict(pairs)
#     order = np.argsort(scores)[::-1]
#     return [(float(scores[i]), hits[i][1]) for i in order]

###############################################################################
# 6) AUGMENT + 7) GENERATE
###############################################################################
def build_prompt(question: str, contexts: List[Dict]) -> str:
    # Keep total context modest to avoid token excess
    dedup = []
    seen = set()
    for c in contexts:
        key = (c["url"], c["chunk"][:120])
        if key not in seen:
            seen.add(key)
            dedup.append(c)

    sources = ""
    for i, c in enumerate(dedup, 1):
        sources += f"\n[SOURCE {i}] {c['url']}\n{c['chunk']}\n"

    return f"""You are a careful analyst.
Answer the user question using ONLY the sources provided.
If the sources are insufficient, say so concisely.
Cite sources as [S{i}] tokens inline.

Question:
{question}

Sources:
{sources}

Final, concise answer with inline citations:"""


def call_llm(prompt: str) -> str:
    """
    Plug in your favorite model provider here.

    # 1) e.g. for Ollama (local):
    # import requests
    # r = requests.post("http://localhost:11434/api/generate",
    #                   json={"model":"llama3", "prompt":prompt, "stream":False})
    # return r.json()["response"]
    
    """
    
    # 2) OpenAI-compatible endpoint:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    resp = client.chat.completions.create(model=GPT_MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0.2)
    
    return resp.choices[0].message.content


###############################################################################
# MAIN
###############################################################################
def main():
    question = "What are Taiwan’s latest AI policies in 2025?"
    # 1) collect corpus
    pages = build_corpus(SEED_URLS, limit_pages=25)

    # 2–4) chunk, embed, index
    vs = VectorStore(ENCODER_NAME)
    vs.add_documents(pages)

    # 4) retrieve
    hits = vs.query(question, top_k=TOP_K)
    # hits = rerank(question, hits)  # (optional)

    contexts = [h[1] for h in hits]
    prompt = build_prompt(question, contexts)
    print("\n================ PROMPT (preview) ================\n")
    print(prompt[:2000], "...\n")

    # 7) generate
    answer = call_llm(prompt)  # <-- wire this up
    print("\n================ ANSWER ================\n")
    print(answer)

if __name__ == "__main__":
    main()
