
from typing import List, Dict, Tuple
import json
import os
from dotenv import load_dotenv

# Dacă ai cheie, poți activa LLM-ul pentru stil conversațional
load_dotenv()
try:
    from openai import OpenAI
    _client = OpenAI() if os.getenv("OPENAI_API_KEY") else None
except Exception:
    _client = None

SYSTEM_RECOMMENDER_CHAT = """Ești un bibliotecar AI prietenos. Primești 
preferințele utilizatorului
și o listă de CANDIDAȚI (titlu, autor, gen, taguri, snippet) recuperați din 
colecția locală.
Reguli:
- Răspunde conversațional în română.
- Recomandă 1–3 cărți din CANDIDAȚI, nu inventa titluri.
- Pentru fiecare: menționează titlul și autorul, 1–2 motive aliniate cu cererea, 
eventual gen/teme.
- Dacă potrivirea e slabă: cere o clarificare (ex. gen, perioadă, vibe) și 
sugerează 1 titlu „cea mai apropiată potrivire”.
- Ton: cald, concis, util.
"""

SYSTEM_CLARIFIER = """Ești un asistent care pune o singură întrebare de 
clarificare,
când cererea e vagă sau rezultatele sunt slabe. Întreabă concis, fără alte 
detalii."""

# Heuristici pentru când să cerem clarificare (tune-uibile)
def _needs_clarification(retrieved: List[Dict]) -> bool:
    if not retrieved:
        return True
    top = float(retrieved[0]["score"]) if retrieved else 0.0
    mean_top3 = sum(float(x["score"]) for x in retrieved[:3]) / max(1, min(3, len(retrieved)))
    return top < 0.30 and mean_top3 < 0.25  # praguri potrivite pentru TF-IDF

def _prepare_candidates(retrieved: List[Dict], max_chars=500, max_items=8):
    cands = []
    for r in retrieved[:max_items]:
        md = r["meta"]
        snippet = r["doc"][:max_chars] + ("..." if len(r["doc"]) > max_chars else "")
        cands.append({
            "title": md.get("title", ""),
            "author": md.get("author", ""),
            "genre": md.get("genre", ""),
            "tags": md.get("tags", ""),
            "score": round(float(r["score"]), 3),
            "snippet": snippet,
        })
    return cands

def ask_chat(query: str, retriever, history: List[Tuple[str, str]] | None = None) -> str:
    """Generăm răspuns conversațional. Dacă nu există cheie OpenAI, folosim fallback local.
    retriever: trebuie să aibă .query(q, k) și să returneze [{score, meta{title,author,genre,tags,info_url}, doc}, ...]
    """
    history = history or []
    retrieved = retriever.query(query, k=10)
    cands = _prepare_candidates(retrieved)
    # dacă nu avem LLM, construim un răspuns simplu (fallback determinist)
    if _client is None:
        if not retrieved:
            return (
                "Nu am găsit potriviri încă. Poți să-mi spui genul preferat sau o temă (ex: prietenie, război, magie)?"
            )
        top = cands[:3]
        bullets = "\n".join([
            f"• **{x['title']}** — {x['author']} (potrivire: {x['score']})" for x in top
        ])
        return (
            f"Pe baza a ce ai spus, iată ce aș recomanda:\n{bullets}\n\n"
            "Dacă vrei, pot rafina dacă îmi spui vibe-ul (optimist, întunecat), perioada sau autorul preferat."
        )
    # LLM disponibil -> clarificare sau recomandare conversațională
    if _needs_clarification(retrieved):
        clar = _client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_CLARIFIER},
                *[{"role": r, "content": c} for (r, c) in history[-6:]],
                {"role": "user", "content": f"Cerere: {query}"},
            ],
        )
        question = clar.choices[0].message.content.strip()
        if retrieved:
            md = retrieved[0]["meta"]
            fallback = f"Între timp, cred că ți-ar putea plăcea **{md.get('title','')}** de {md.get('author','')} (cea mai apropiată potrivire)."
            return f"{question}\n\n{fallback}"
        return question
    user_payload = {"query": query, "candidates": cands}
    chat = _client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_RECOMMENDER_CHAT},
            *[{"role": r, "content": c} for (r, c) in history[-6:]],
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )
    return chat.choices[0].message.content.strip()