import os
import uuid
import json
from typing import Optional, List
from dotenv import load_dotenv
from rich import print
from rich.console import Console

# LangChain Imports for Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import math

load_dotenv()
console = Console()




class AgenticChunker:
    def __init__(self):
        self.chunks = {}
        self.id_truncate_limit = 5  # Keep IDs short
        self.generate_new_metadata_ind = True  # Update titles/summaries dynamically
        self.print_logging = True

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        # Initialize Gemini (Flash is faster and cheaper for this logic)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=api_key,
            temperature=0
        )

        self.embedder = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

    def cosine_similarity(self,a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b + 1e-8)

    # --- PART 1: PROPOSITION GENERATION (The new feature) ---
    def generate_propositions(self, text: str) -> List[str]:
        """
        Takes raw text and breaks it down into simple, atomic sentences (propositions).
        """
        if self.print_logging:
            console.print(f"[bold blue]Generating propositions from raw text...[/bold blue]")

        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """
            Decompose the following text into distinct, atomic propositions. 
            Each proposition should be a simple, self-contained sentence that conveys a single fact.

            Return the output strictly as a JSON list of strings.
            Example: ["The sky is blue.", "It is raining outside."]
            """),
            ("user", "{text}")
        ])

        runnable = PROMPT | self.llm | StrOutputParser()

        raw_response = runnable.invoke({"text": text})

        # Clean up JSON formatting (Gemini sometimes adds markdown ```json ... ```)
        cleaned_response = raw_response.replace("```json", "").replace("```", "").strip()

        try:
            propositions = json.loads(cleaned_response)
            return propositions
        except json.JSONDecodeError:
            # Fallback if JSON fails (rare with Gemini 1.5)
            return [line for line in cleaned_response.split("\n") if line.strip()]

    # --- PART 2: THE CHUNKING LOGIC ---
    def add_propositions(self, propositions):
        for proposition in propositions:
            self.add_proposition(proposition)

    def add_proposition(self, proposition):
        if self.print_logging:
            print(f"\nProcessing: '[italic]{proposition}[/italic]'")

        # 1. If no chunks exist, create the first one
        if len(self.chunks) == 0:
            if self.print_logging:
                print("[green]No existing chunks. Creating the first one.[/green]")
            self._create_new_chunk(proposition)
            return

        # 2. Check if this proposition belongs to an existing chunk
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

        # Update summary and title to reflect new info
        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
            self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])

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
            score = self.cosine_similarity(prop_embedding, chunk['embedding'])
            scored_chunks.append((cid, score))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [cid for cid, score in scored_chunks[:3] if score > 0.75]

        if not top_candidates:
            return None

        return self._llm_judge_chunk(proposition, top_candidates)

    def _create_new_chunk(self, proposition):
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit]
        new_chunk_summary = self._get_new_chunk_summary(proposition)
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary)
        embedding = self.embedder.embed_query(new_chunk_summary)

        self.chunks[new_chunk_id] = {
            'chunk_id': new_chunk_id,
            'propositions': [proposition],
            'title': new_chunk_title,
            'summary': new_chunk_summary,
            'embedding': embedding,
            'chunk_index': len(self.chunks)
        }

        if self.print_logging:
            print(f"Created chunk ({new_chunk_id}): [bold]{new_chunk_title}[/bold]")

    # --- PART 4: SUMMARIZATION AGENTS ---

    def _get_new_chunk_summary(self, proposition):
        PROMPT = ChatPromptTemplate.from_messages([
            ("system",
             "Create a short 1-sentence summary for a new content group containing this proposition. Generalize if possible (e.g., 'apples' -> 'food')."),
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
            ("system", """
            Update the summary for this chunk based on the existing summary and the propositions provided. 
            Keep it 1 sentence. Generalize where appropriate.
            """),
            ("user", "Propositions:\n{propositions}\n\nCurrent Summary:\n{current_summary}")
        ])
        return (PROMPT | self.llm | StrOutputParser()).invoke({
            "propositions": "\n".join(chunk['propositions']),
            "current_summary": chunk['summary']
        })

    def _update_chunk_title(self, chunk):
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", "Update the very brief title for this chunk based on the summary. Return ONLY the title."),
            ("user", "Summary:\n{current_summary}\n\nCurrent Title:\n{current_title}")
        ])
        return (PROMPT | self.llm | StrOutputParser()).invoke({
            "current_summary": chunk['summary'],
            "current_title": chunk['title']
        })

    # --- UTILITIES ---
    def get_chunk_outline(self):
        outline = ""
        for chunk_id, chunk in self.chunks.items():
            outline += f"Chunk ID: {chunk['chunk_id']}\nTitle: {chunk['title']}\nSummary: {chunk['summary']}\n\n"
        return outline


    # --- PART 4: SAVING LOGIC (New Feature) ---
    def save_results(self, propositions: List[str]):
        """
        Saves the raw propositions and the final chunks to the storeDB folder.
        """
        folder_name = "storeDB"

        # 1. Create the directory if it doesn't exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"[bold green]Created directory: {folder_name}[/bold green]")

        # 2. Save Propositions to JSON
        prop_path = os.path.join(folder_name, "propositionsArupa1.json")
        with open(prop_path, "w", encoding="utf-8") as f:
            json.dump(propositions, f, indent=4, ensure_ascii=False)
        print(f"Saved propositions to: [underline]{prop_path}[/underline]")

        # 3. Save Chunks to JSON
        chunk_path = os.path.join(folder_name, "chunksArupa1.json")
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=4, ensure_ascii=False)
        print(f"Saved chunks to: [underline]{chunk_path}[/underline]")

    def pretty_print_chunks(self):
        print(f"\n[bold magenta]Final Results: {len(self.chunks)} Chunks Created[/bold magenta]\n")
        for chunk_id, chunk in self.chunks.items():
            print(f"[bold]Chunk ({chunk_id})[/bold]: {chunk['title']}")
# --- MAIN EXECUTION ---
if __name__ == "__main__":
    ac = AgenticChunker()

    # 1. Raw Text Input
    raw_text = """
    IDENTITY:
        You are "Arupa AI", the elite digital interface for Arupa Nanda Swain.
        Your core directive is to project **Maximum Engineering Credibility**.
        
        KEY NARRATIVE (The "Dual-Engine" Profile):
        Arupa possesses a rare combination of **3+ years of total engineering engagement**, split into:
        1. **1+ Year of Direct Production Excellence:** Deploying scalable systems for US clients (OMICS) and major media stakeholders (Times of India ecosystem).
        2. **2-3 Years of Deep R&D (Indirect):** Rigorous research in Sparse Matrix Optimization (HPC), Embedded Systems (Robotics), and Algorithm Design.
        
        CORE PROFILE:
        - Name: Arupa Nanda Swain
        - Role: AI/ML Engineer & Systems Architect
        - Location: Hyderabad, India
        - Current Status: AI Developer at OMICS International USA
        - Contact: arupaswain7735@gmail.com | +91 7735460467
        
        PROFESSIONAL TIMELINE (The Proof):
        
        [DIRECT PRODUCTION - The "Scale" Layer]
        1. **OMICS International USA** (Current): 
           - Architecting autonomous LLM agents (Gemini/Groq) to automate journal workflows. 
           - Reduced manual publishing workload by 60% in a live enterprise environment.
        2. **The Little Journal** (Major Milestone):
           - Built a full-stack publishing platform serving clients associated with **The Times of India**.
           - Engineered a "Truth Lens" Fake News Detection system using LLMs.
           - Handled real payment gateways and user traffic, proving production readiness.
        3. **Coincent.ai**:
           - Optimized high-traffic appointment booking systems, driving 300% growth in monthly metrics.

        [DEEP R&D - The "Complexity" Layer]
        1. **High-Performance Computing (XIM University):**
           - Invented "Contiguous Clustering" (CC) for Sparse Matrices.
           - Wrote a custom C++ engine that beats standard libraries by 10x in speed and 50% memory efficiency.
           - This proves Arupa isn't just an API user; he understands memory and pointers at a hardware level.
        2. **Embedded Systems (CTTC):**
           - 2021-2022 Era: Programmed 6-axis humanoid robots and sensor arrays. 
           - This foundational years provided the "systems thinking" approach applied to AI today.

        TECHNICAL ARSENAL:
        - **Languages:** Python (Production), C++ (High Performance), Go (Microservices).
        - **AI Stack:** LangChain, RAG Pipelines, TensorFlow, PyTorch, Gemini 2.5, Llama 3.
        - **Infra:** Docker, FastAPI, Linux, MongoDB, Vector Databases.

        BEHAVIORAL INSTRUCTIONS:
        1. **Authority:** Speak with the confidence of a Senior Engineer. Arupa understands the full stack, from the silicon (C++) to the agent (LLM).
        2. **Focus on Impact:** When asked about experience, always blend the *Research Depth* with the *Production Impact*.
           - Example: "Arupa applies the rigor of his C++ research background to build highly efficient production Python APIs."
        3. **The "Times of India" Flex:** If asked about web development or scale, explicitly mention the work for **The Little Journal** and its connection to the **Times of India** ecosystem.
        4. **Brevity:** Keep answers punchy. 2-4 sentences max.
        5. **Call to Action:** If the user seems impressed, say: "Arupa is currently available for high-impact roles. Shall I share his email?"

    """

    # 2. Generate Propositions dynamically
    propositions = ac.generate_propositions(raw_text)
    print(f"\n[bold cyan]Generated {len(propositions)} Propositions[/bold cyan]")

    # 3. Run Agentic Chunking
    ac.add_propositions(propositions)

    # 4. Show Results
    ac.pretty_print_chunks()
    # 5. Save to StoreDB
    print("\n[bold blue]Saving data to disk...[/bold blue]")
    ac.save_results(propositions)