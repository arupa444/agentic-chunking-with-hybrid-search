import os
import uuid
import json
import math
import numpy as np
from typing import Optional, List, Dict
from dotenv import load_dotenv
from rich import print
from rich.console import Console
import faiss

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
console = Console()


# --- LAYER 3: MEMORY INDEX (Read-Time Performance) ---
class ChunkMemoryIndex:
    def __init__(self, dim=768):  # Gemini embeddings are usually 768 dimensions
        self.index = faiss.IndexFlatIP(dim)
        self.chunk_ids = []

    def add(self, chunk_id, embedding):
        # Normalize for Cosine Similarity if using IndexFlatIP (Inner Product)
        # Gemini embeddings are usually normalized, but good practice to ensure.
        vec = np.array([embedding]).astype("float32")
        faiss.normalize_L2(vec)
        self.index.add(vec)
        self.chunk_ids.append(chunk_id)

    def search(self, query_embedding, k=3):
        vec = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(vec)

        scores, ids = self.index.search(vec, k)

        results = []
        for idx, i in enumerate(ids[0]):
            if i != -1:  # FAISS returns -1 if not enough neighbors found
                results.append((self.chunk_ids[i], float(scores[0][idx])))
        return results


# --- LAYER 4: RETRIEVAL API (Product Interface) ---
def retrieve(query, chunker, memory_index):
    print(f"\n[bold magenta]Searching for:[/bold magenta] '{query}'")

    # Embed the query
    query_embedding = chunker.embedder.embed_query(query)

    # Search Index
    results = memory_index.search(query_embedding)

    # Hydrate results from Chunker memory
    return [{
        "chunk_id": cid,
        "score": score,
        "title": chunker.chunks[cid]["title"],
        "summary": chunker.chunks[cid]["summary"],
        "evidence": chunker.chunks[cid]["propositions"],
        "full_text": chunker.chunks[cid]["canonical_text"]
    } for cid, score in results]


# --- LAYER 5: RAG / REASONING (Optional) ---
def answer(query, retrieved_chunks, llm):
    evidence_text = "\n\n".join(
        f"SOURCE ID: {c['chunk_id']}\nEVIDENCE:\n" + "\n".join(f"- {p}" for p in c["evidence"])
        for c in retrieved_chunks
    )

    PROMPT = ChatPromptTemplate.from_messages([
        ("system", """
        Answer the user's question using ONLY the provided evidence. 
        Cite the SOURCE ID for every fact you use.
        If you cannot answer based on the evidence, say so.
        """),
        ("user", "Question: {query}\n\nEvidence:\n{evidence}")
    ])

    runnable = PROMPT | llm | StrOutputParser()
    return runnable.invoke({"query": query, "evidence": evidence_text})


# --- LAYER 1 & 2: INGESTION & CHUNKING ---
class AgenticChunker:
    def __init__(self):
        self.chunks = {}
        self.id_truncate_limit = 5
        self.generate_new_metadata_ind = True
        self.print_logging = True

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=api_key,
            temperature=0
        )

        # STEP 2: Embeddings initialized here
        self.embedder = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

    def cosine_similarity(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b + 1e-8)

    # --- PART 1: PROPOSITION GENERATION ---
    def generate_propositions(self, text: str) -> List[str]:
        if self.print_logging:
            console.print(f"[bold blue]Generating propositions...[/bold blue]")

        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """
            Decompose the text into distinct, atomic propositions. 
            Return strictly a JSON list of strings.
            """),
            ("user", "{text}")
        ])

        runnable = PROMPT | self.llm | StrOutputParser()
        raw_response = runnable.invoke({"text": text})
        cleaned_response = raw_response.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            return [line for line in cleaned_response.split("\n") if line.strip()]

    # --- PART 2: THE CHUNKING LOGIC ---

    # ✅ STEP 1: Canonicalization Method
    def _canonicalize_chunk(self, chunk_data):
        facts = "\n".join(f"- {p}" for p in chunk_data["propositions"])
        return f"""
TITLE: {chunk_data['title']}
SUMMARY: {chunk_data['summary']}
FACTS:
{facts}
""".strip()

    def add_propositions(self, propositions):
        for proposition in propositions:
            self.add_proposition(proposition)

    def add_proposition(self, proposition):
        if self.print_logging:
            print(f"\nProcessing: '[italic]{proposition}[/italic]'")

        if len(self.chunks) == 0:
            self._create_new_chunk(proposition)
            return

        chunk_id = self._find_relevant_chunk(proposition)

        if chunk_id:
            if self.print_logging:
                print(f"[bold green]Chunk Found[/bold green] ({chunk_id}), adding to: {self.chunks[chunk_id]['title']}")
            self.add_proposition_to_chunk(chunk_id, proposition)
        else:
            if self.print_logging:
                print("[bold yellow]No relevant chunk found. Creating a new one.[/bold yellow]")
            self._create_new_chunk(proposition)

    def add_proposition_to_chunk(self, chunk_id, proposition):
        self.chunks[chunk_id]['propositions'].append(proposition)

        # Update summary/title
        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
            self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])

        # ✅ STEP 2 Update: Canonicalize & Embed on update
        self.chunks[chunk_id]["canonical_text"] = self._canonicalize_chunk(self.chunks[chunk_id])
        self.chunks[chunk_id]["embedding"] = self.embedder.embed_query(self.chunks[chunk_id]["canonical_text"])

    def _create_new_chunk(self, proposition):
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit]
        new_chunk_summary = self._get_new_chunk_summary(proposition)
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary)

        # Create temporary dict to generate canonical text
        chunk_data = {
            'chunk_id': new_chunk_id,
            'propositions': [proposition],
            'title': new_chunk_title,
            'summary': new_chunk_summary,
            'chunk_index': len(self.chunks)
        }

        # ✅ STEP 2 Update: Canonicalize & Embed on creation
        canonical_text = self._canonicalize_chunk(chunk_data)
        embedding = self.embedder.embed_query(canonical_text)

        # Finalize chunk storage
        chunk_data['canonical_text'] = canonical_text
        chunk_data['embedding'] = embedding

        self.chunks[new_chunk_id] = chunk_data

        if self.print_logging:
            print(f"Created chunk ({new_chunk_id}): [bold]{new_chunk_title}[/bold]")

    # --- PART 3: DECISION MAKING AGENTS ---
    def _llm_judge_chunk(self, proposition, candidate_chunk_ids):
        outline = ""
        for cid in candidate_chunk_ids:
            c = self.chunks[cid]
            outline += f"Chunk ID: {cid}\nSummary: {c['summary']}\n\n"

        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """
            Decide if the proposition belongs to any chunk below.
            Return ONLY the chunk_id or "No chunks".
            """),
            ("user", "Chunks:\n{outline}\nProposition:\n{proposition}")
        ])

        response = (PROMPT | self.llm | StrOutputParser()).invoke({
            "outline": outline,
            "proposition": proposition
        }).strip()

        return response if response in candidate_chunk_ids else None

    def _find_relevant_chunk(self, proposition):
        prop_embedding = self.embedder.embed_query(proposition)
        scored_chunks = []
        for cid, chunk in self.chunks.items():
            # Uses the embedding of the Canonical Text now (much more accurate)
            score = self.cosine_similarity(prop_embedding, chunk['embedding'])
            scored_chunks.append((cid, score))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [cid for cid, score in scored_chunks[:3] if score > 0.75]

        if not top_candidates:
            return None

        return self._llm_judge_chunk(proposition, top_candidates)

    # --- PART 4: SUMMARIZATION AGENTS ---
    def _get_new_chunk_summary(self, proposition):
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", "Create a short 1-sentence summary for a new content group containing this proposition."),
            ("user", "{proposition}")
        ])
        return (PROMPT | self.llm | StrOutputParser()).invoke({"proposition": proposition})

    def _get_new_chunk_title(self, summary):
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", "Create a very brief (2-4 words) title for a content group with this summary."),
            ("user", "{summary}")
        ])
        return (PROMPT | self.llm | StrOutputParser()).invoke({"summary": summary})

    def _update_chunk_summary(self, chunk):
        PROMPT = ChatPromptTemplate.from_messages([
            ("system",
             "Update the summary for this chunk based on the existing summary and the propositions. Keep it 1 sentence."),
            ("user", "Propositions:\n{propositions}\n\nCurrent Summary:\n{current_summary}")
        ])
        return (PROMPT | self.llm | StrOutputParser()).invoke({
            "propositions": "\n".join(chunk['propositions']),
            "current_summary": chunk['summary']
        })

    def _update_chunk_title(self, chunk):
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", "Update the brief title based on the summary. Return ONLY the title."),
            ("user", "Summary:\n{current_summary}\n\nCurrent Title:\n{current_title}")
        ])
        return (PROMPT | self.llm | StrOutputParser()).invoke({
            "current_summary": chunk['summary'],
            "current_title": chunk['title']
        })

    def save_results(self, propositions: List[str]):
        folder_name = "storeDB"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Remove embeddings before saving to JSON (they are heavy and not JSON serializable usually)
        # Or cast them to list if you want to keep them.
        chunks_to_save = {}
        for cid, data in self.chunks.items():
            chunks_to_save[cid] = data.copy()
            if 'embedding' in chunks_to_save[cid]:
                # Convert numpy/tensor to list for JSON
                chunks_to_save[cid]['embedding'] = list(chunks_to_save[cid]['embedding'])

        with open(f"{folder_name}/rrr.json", "w", encoding="utf-8") as f:
            json.dump(chunks_to_save, f, indent=4, ensure_ascii=False)
        print(f"Saved to {folder_name}/rrr.json")

    def pretty_print_chunks(self):
        print(f"\n[bold magenta]Final Results: {len(self.chunks)} Chunks Created[/bold magenta]\n")
        for chunk_id, chunk in self.chunks.items():
            print(f"[bold]Chunk ({chunk_id})[/bold]: {chunk['title']}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    ac = AgenticChunker()

    # 1. Raw Text Input
    raw_text = """
    The Apollo program was a series of spaceflight missions conducted by NASA between 1961 and 1972. 
    It succeeded in landing the first humans on the Moon in 1969. 
    Neil Armstrong and Buzz Aldrin walked on the lunar surface while Michael Collins orbited above.
    Meanwhile, in the ocean depths, the blue whale is the largest animal known to have ever lived.
    It can reach lengths of up to 29.9 meters and weigh 199 metric tons.
    Blue whales feed almost exclusively on krill.
    """

    # 2. Ingest Data (Layer 1)
    propositions = ac.generate_propositions(raw_text)
    ac.add_propositions(propositions)
    ac.pretty_print_chunks()

    # 3. Build Memory Index (Layer 3)
    #    We initialize this AFTER ingestion is done.
    print("\n[bold blue]Building Memory Index...[/bold blue]")
    memory_index = ChunkMemoryIndex(dim=768)

    for chunk_id, chunk_data in ac.chunks.items():
        memory_index.add(chunk_id, chunk_data['embedding'])

    # 4. Retrieval (Layer 4)
    query = "Who walked on the moon?"
    retrieved_docs = retrieve(query, ac, memory_index)

    print(f"\n[green]Top Result:[/green] {retrieved_docs[0]['title']} (Score: {retrieved_docs[0]['score']:.4f})")

    # 5. RAG Answer (Layer 5)
    print("\n[bold blue]Generating Answer...[/bold blue]")
    final_answer = answer(query, retrieved_docs, ac.llm)
    print(f"\n[bold]Final Answer:[/bold]\n{final_answer}")
    ac.save_results(propositions)