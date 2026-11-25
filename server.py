"""RobustRAG Product API - FastAPI Service.

Exposes RESTful endpoints for:
1. Live Interactive RAG Pipeline (Query -> Retrieval -> Noise Injection -> Reranking -> Answer Generation)
2. Evaluation & Robustness Stress Testing (Runs across multiple noise levels with metrics)
3. Corpus Management (Add documents, query corpus)
4. Historical Experiment Run Insights (Metrics, figures, summaries)
"""

import os
import sys
import time
import json
import uuid
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure src is in python path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from robustrag.corpus import DocumentCorpus, Passage
from robustrag.retrieval import BM25Retriever
from robustrag.distractors import DistractorGenerator
from robustrag.reranking import CrossEncoderReranker, MockReranker, order_passages_unranked
from robustrag.generation import LocalHuggingFaceGenerator, GroqGenerator, MockGenerator
from robustrag.metrics import evaluate_qa_prediction
from robustrag.config import GeneratorConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("robustrag.api")

app = FastAPI(
    title="RobustRAG Enterprise Platform",
    description="Production-grade RAG Reliability, Noise-Resistance & Benchmarking Suite",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State / Singletons
corpus = DocumentCorpus()
retriever: Optional[BM25Retriever] = None
distractor_gen: Optional[DistractorGenerator] = None
reranker_model: Optional[CrossEncoderReranker] = None

# Lazy/Cached generator instance
generators_cache: Dict[str, Any] = {}

def get_or_create_generator(backend: str = "mock", model_name: str = "mock-generator"):
    key = f"{backend}_{model_name}"
    if key in generators_cache:
        return generators_cache[key]
    
    cfg = GeneratorConfig(backend=backend, model_name=model_name, temperature=0.0, max_new_tokens=64)
    if backend == "local_hf":
        gen = LocalHuggingFaceGenerator(cfg)
    elif backend == "groq":
        gen = GroqGenerator(cfg)
    else:
        gen = MockGenerator(cfg)
    generators_cache[key] = gen
    return gen

def initialize_default_corpus():
    global corpus, retriever, distractor_gen
    sample_file = BASE_DIR / "data" / "squad_sample_fallback.json"
    corpus = DocumentCorpus()
    if sample_file.exists():
        with open(sample_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            from robustrag.data_loader import QAExample
            examples = [
                QAExample(
                    id=item.get("id", f"q_{i}"),
                    title=item.get("title", "Article"),
                    context=item.get("context", ""),
                    question=item.get("question", ""),
                    answers=item.get("answers", {}).get("text", []) if isinstance(item.get("answers"), dict) else item.get("answers", [])
                )
                for i, item in enumerate(data)
            ]
            corpus.build_from_examples(examples)
            logger.info("Loaded default corpus with %d passages.", len(corpus.passages))
    else:
        # Fallback passages
        p1 = Passage(passage_id="doc_1", title="Transformers in NLP", text="Transformers are deep learning models introduced in 2017 utilizing self-attention mechanisms to process sequential text.", char_length=125)
        p2 = Passage(passage_id="doc_2", title="Retrieval-Augmented Generation", text="RAG combines external document retrieval with generative neural language models to ground responses in verifiable source facts.", char_length=124)
        corpus.passages[p1.passage_id] = p1
        corpus.passages[p2.passage_id] = p2
    
    retriever = BM25Retriever(corpus)
    distractor_gen = DistractorGenerator(corpus=corpus, retriever=retriever, random_seed=42)

# Initialize corpus on startup
initialize_default_corpus()


# --- Pydantic Request/Response Models ---

class QueryRequest(BaseModel):
    query: str = Field(..., example="What mechanism do transformers utilize?")
    gold_answer: Optional[str] = Field(default="", example="self-attention")
    gold_context: Optional[str] = Field(default="", example="")
    k_distractors: int = Field(default=2, ge=0, le=10)
    distractor_type: str = Field(default="hard", description="random or hard")
    use_reranker: bool = Field(default=True)
    generator_backend: str = Field(default="mock", description="mock, local_hf, or groq")
    generator_model: str = Field(default="google/flan-t5-small")

class DocumentItem(BaseModel):
    id: str
    title: str
    text: str
    score: Optional[float] = None
    is_gold: bool = False
    is_distractor: bool = False

class QueryResponse(BaseModel):
    answer: str
    latency_ms: float
    documents_used: List[DocumentItem]
    reranked: bool
    distractor_count: int
    metrics: Optional[Dict[str, float]] = None

class AddDocumentRequest(BaseModel):
    title: str
    text: str

class BenchmarkRequest(BaseModel):
    questions_count: int = Field(default=5, ge=1, le=25)
    noise_levels: List[int] = Field(default=[0, 1, 3])
    generator_backend: str = Field(default="mock")


# --- API Routes ---

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "RobustRAG Enterprise Platform",
        "corpus_size": len(corpus.passages),
        "available_backends": ["mock", "local_hf", "groq"]
    }

@app.get("/api/corpus")
def get_corpus(limit: int = 20):
    all_ids = corpus.get_all_passage_ids()[:limit]
    items = []
    for pid in all_ids:
        p = corpus.get_passage(pid)
        if p:
            items.append({"passage_id": p.passage_id, "title": p.title, "text": p.text[:200] + ("..." if len(p.text) > 200 else "")})
    return {
        "total_documents": len(corpus.passages),
        "sample": items
    }

@app.post("/api/corpus/add")
def add_to_corpus(doc: AddDocumentRequest):
    global retriever, distractor_gen
    new_id = f"doc_{uuid.uuid4().hex[:8]}"
    passage = Passage(passage_id=new_id, title=doc.title, text=doc.text, char_length=len(doc.text))
    corpus.passages[new_id] = passage
    # Reindex
    retriever = BM25Retriever(corpus)
    distractor_gen = DistractorGenerator(corpus=corpus, retriever=retriever, random_seed=42)
    return {"message": "Document indexed successfully", "passage_id": new_id, "corpus_size": len(corpus.passages)}

@app.post("/api/query", response_model=QueryResponse)
def execute_query(req: QueryRequest):
    start_time = time.time()
    
    # 1. Identify or retrieve base gold context
    candidate_passages: List[Passage] = []
    is_custom_gold = bool(req.gold_context.strip())
    
    if is_custom_gold:
        gold_passage = Passage(passage_id="gold_custom", title="Target Evidence", text=req.gold_context.strip(), char_length=len(req.gold_context.strip()))
        candidate_passages.append(gold_passage)
    else:
        # Retrieve best matching passage from corpus
        retrieved = retriever.retrieve(req.query, top_k=1)
        if retrieved:
            gold_passage = retrieved[0].passage
            candidate_passages.append(gold_passage)
        else:
            gold_passage = Passage(passage_id="fallback", title="Corpus Evidence", text="No documents retrieved.", char_length=23)
            candidate_passages.append(gold_passage)
            
    gold_id = gold_passage.passage_id

    # 2. Inject Distractors (Noise Injection)
    if req.k_distractors > 0:
        if req.distractor_type == "hard":
            distractors, _ = distractor_gen.get_hard_distractors(
                question_id=gold_id,
                question_text=req.query,
                reference_answers=[req.gold_answer] if req.gold_answer else [],
                count=req.k_distractors
            )
        else:
            distractors = distractor_gen.get_random_distractors(
                question_id=gold_id,
                count=req.k_distractors
            )
        candidate_passages.extend(distractors)

    # 3. Ordering: Reranking vs Unranked Baseline
    scores_dict: Dict[str, float] = {}
    if req.use_reranker:
        global reranker_model
        if reranker_model is None:
            reranker_model = CrossEncoderReranker()
        
        scored_pairs = reranker_model.score_passages(req.query, candidate_passages)
        ordered_passages = [p for p, _ in scored_pairs]
        scores_dict = {p.passage_id: float(sc) for p, sc in scored_pairs}
    else:
        # Deterministic unranked shuffle baseline
        ordered_passages = order_passages_unranked(candidate_passages, question_id=req.query, random_seed=42)

    # 4. Answer Generation
    generator = get_or_create_generator(req.generator_backend, req.generator_model)
    answer = generator.generate_answer(
        question=req.query,
        passages=ordered_passages
    )

    latency_ms = (time.time() - start_time) * 1000

    # 5. Evaluate if gold answer provided
    eval_metrics = None
    if req.gold_answer.strip():
        f1_val, em_val = evaluate_qa_prediction(answer, [req.gold_answer.strip()])
        eval_metrics = {"f1": f1_val, "exact_match": em_val}

    docs_output = [
        DocumentItem(
            id=p.passage_id,
            title=p.title,
            text=p.text,
            score=scores_dict.get(p.passage_id),
            is_gold=(p.passage_id == gold_id),
            is_distractor=(p.passage_id != gold_id)
        )
        for p in ordered_passages
    ]

    return QueryResponse(
        answer=answer,
        latency_ms=round(latency_ms, 2),
        documents_used=docs_output,
        reranked=req.use_reranker,
        distractor_count=req.k_distractors,
        metrics=eval_metrics
    )

@app.get("/api/experiments/runs")
def list_experiment_runs():
    runs_dir = BASE_DIR / "outputs" / "runs"
    runs = []
    if runs_dir.exists():
        for run_folder in runs_dir.iterdir():
            if run_folder.is_dir():
                summary_file = run_folder / "summary.json"
                metadata_file = run_folder / "metadata.json"
                meta = {}
                summary = {}
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            meta = json.load(f)
                    except Exception:
                        pass
                if summary_file.exists():
                    try:
                        with open(summary_file, "r") as f:
                            summary = json.load(f)
                    except Exception:
                        pass
                runs.append({
                    "id": run_folder.name,
                    "title": meta.get("experiment_name", run_folder.name),
                    "created_at": meta.get("start_time", ""),
                    "status": meta.get("status", "completed"),
                    "summary": summary
                })
    return {"runs": runs}

@app.get("/api/experiments/runs/{run_id}")
def get_run_details(run_id: str):
    run_folder = BASE_DIR / "outputs" / "runs" / run_id
    if not run_folder.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    
    result = {"id": run_id}
    for filename in ["summary.json", "metadata.json", "scores.csv"]:
        fpath = run_folder / filename
        if fpath.exists():
            if filename.endswith(".json"):
                with open(fpath, "r") as f:
                    result[filename.replace(".json", "")] = json.load(f)
            elif filename.endswith(".csv"):
                import pandas as pd
                df = pd.read_csv(fpath)
                result["scores"] = df.to_dict(orient="records")

    report_path = run_folder / "report.md"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            result["report_markdown"] = f.read()

    return result


# Mount modern static Web App if static directory exists
static_path = BASE_DIR / "static"
if static_path.exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
