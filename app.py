import asyncio
import json
import os
import time
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, parse_qs

import httpx
import uvicorn
from google import genai
from bs4 import BeautifulSoup
import aiosqlite
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class ResearchRequest(BaseModel):
    query: str
    config: Dict[str, Any]

# --- Configuration Class ---
@dataclass
class ResearchConfig:
    """Configuration settings for the research tool."""
    depth: int = 2              
    breadth: int = 3            
    max_content_length: int = 60000  
    search_timeout: int = 30    
    citation_mode: bool = True  
    model_name: str = "gemini-3.1-flash-lite-preview" # Default fast model

# --- FastAPI App Setup ---
# The api/index.py imports this `app` variable directly.
app = FastAPI(title="Deep Research API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# --- Database Setup ---
DATABASE = "/tmp/deep_research.db" if os.environ.get("VERCEL") else "deep_research.db"

async def init_db():
    """Initializes the SQLite database."""
    async with aiosqlite.connect(DATABASE) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS research_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                query TEXT,
                report TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                history TEXT,
                updated_at DATETIME
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS content_cache (
                url TEXT PRIMARY KEY,
                content TEXT,
                title TEXT,
                updated_at DATETIME
            )
        """)
        await db.commit()
    logger.info("Database initialized.")

async def get_db():
    async with aiosqlite.connect(DATABASE) as db:
        yield db

class ChatSession:
    """Manages conversation history with Gemini API using new google-genai format."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history = []
        self.db_path = DATABASE
        
    async def load_history(self):
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute("SELECT history FROM chat_sessions WHERE session_id=?", (self.session_id,)) as cursor:
                    row = await cursor.fetchone()
                    if row and row["history"]:
                        try:
                            self.history = json.loads(row["history"])
                        except json.JSONDecodeError:
                            self.history = []
                    else:
                        self.history = []
        except Exception as e:
            logger.error(f"Failed to load chat history: {e}")

    async def save_history(self):
        try:
            async with aiosqlite.connect(self.db_path) as db:
                filtered_history = [m for m in self.history if m.get("role") in ("user", "model")]
                history_json = json.dumps(filtered_history)
                await db.execute("REPLACE INTO chat_sessions (session_id, history, updated_at) VALUES (?, ?, ?)",
                                (self.session_id, history_json, datetime.now()))
                await db.commit()
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "parts": [{"text": content}]})
        # Save asynchronously without blocking
        from asyncio import create_task
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.save_history())
        except RuntimeError:
            pass

    def get_history(self):
        return self.history

class DeepResearchTool:
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.client = None
        self.worker_model = "gemini-3.1-flash-lite-preview"
        self.orchestrator_model = "gemini-3.1-pro-preview"
        self.http_client = httpx.AsyncClient(timeout=config.search_timeout, follow_redirects=True)

    def configure_api(self, api_key: Optional[str] = None):
        if api_key:
            api_key = api_key.strip()
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") 

        if not api_key:
            raise ValueError("No API key provided.")

        # STRICTLY use the standard genai Client
        self.client = genai.Client(api_key=api_key)
        logger.info(f"Gemini API configured: Orchestrator-Worker Hybrid Architecture enabled.")

    async def search_web(self, query: str, num_results: int = None) -> List[Dict[str, str]]:
        if num_results is None:
            num_results = self.config.breadth

        results = []
        try:
            # DuckDuckGo HTML parsing
            search_url = "https://html.duckduckgo.com/html/"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            }
            data = {"q": query}
            
            response = await self.http_client.post(search_url, data=data, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                results_elements = soup.find_all('div', class_='result')
                
                for item in results_elements:
                    title_elem = item.find('a', class_='result__url')
                    snippet_elem = item.find('a', class_='result__snippet')
                    
                    if title_elem and snippet_elem:
                        title = title_elem.text.strip()
                        raw_url = title_elem.get('href', '')
                        
                        if raw_url.startswith('//duckduckgo.com/l/?'):
                            parsed = parse_qs(urlparse(raw_url).query)
                            real_url = parsed.get('uddg', [raw_url])[0]
                        else:
                            real_url = raw_url
                            
                        snippet = snippet_elem.text.strip()
                        results.append({
                            "title": title,
                            "url": real_url,
                            "snippet": snippet
                        })
                        
                        if len(results) >= num_results:
                            break
                            
            else:
                logger.error(f"DuckDuckGo error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"DuckDuckGo search exception: {e}")
            
        if not results:
            results.append({"title": "No results found", "url": "", "snippet": f"Could not find information on DuckDuckGo for: {query}"})
            
        return results

    async def _generate_with_fallback(self, contents: Any, role: str = "worker", **kwargs) -> Any:
        if not self.client:
            raise ValueError("Gemini API not configured.")

        # Worker role exclusively uses Flash Lite. Orchestrator uses Pro.
        primary_model = self.worker_model if role == "worker" else self.orchestrator_model
        fallback_model = self.worker_model
        
        try:
            return await self.client.aio.models.generate_content(
                model=primary_model,
                contents=contents,
                **kwargs
            )
        except Exception as e:
            logger.warning(f"Primary model {primary_model} failed: {e}. Falling back to {fallback_model}")
            if primary_model != fallback_model:
                try:
                    return await self.client.aio.models.generate_content(
                        model=fallback_model,
                        contents=contents,
                        **kwargs
                    )
                except Exception as fallback_err:
                    logger.error(f"Fallback model {fallback_model} also failed: {fallback_err}")
                    raise fallback_err
            else:
                raise e

    async def generate_search_queries(self, content: str) -> List[str]:
        # Generate as many queries as the LLM deems necessary to cover missing knowledge branches, up to 10.
        prompt = f"""Read the following research content. Identify all areas that require further investigation, and determine exactly how many specific DuckDuckGo search queries are needed to deeply map out the missing details.
Generate the optimal number of queries (between 1 and 10) to fetch the required information.
Return ONLY a valid JSON array of strings, where each string is a search query. Do not include markdown codeblocks or other text.
CONTENT: {content[:10000]}"""

        try:
            # Always route standard recursive tasks to the ultra-fast worker
            response = await self._generate_with_fallback(contents=prompt, role="worker")
            # Cleanup output and parse JSON
            text = response.text.strip()
            text = re.sub(r'```[a-z]*\n(.*?)\n```', r'\1', text, flags=re.DOTALL).strip()
            queries = json.loads(text)
            if isinstance(queries, list):
                 return queries[:10]  # Hard cap at 10 simultaneous spawned branches
        except Exception as e:
            logger.error(f"Query generation error: {e}")
            
        # Fallback query parsing
        return ["Follow up research query"]

    async def chat_with_content(
        self,
        user_query: str,
        session: ChatSession,
        search_results: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        
        search_context = ""
        sources = []
        if search_results:
            search_context = "Search results:\n"
            for i, result in enumerate(search_results, 1):
                if result['url']:
                    search_context += f"[{i}] {result['title']}\nURL: {result['url']}\nSnippet: {result['snippet']}\n\n"
                    sources.append({"index": i, "title": result['title'], "url": result['url']})

        # Compile Chat History
        history = session.get_history()
        history_str = ""
        if history:
            history_str = "PREVIOUS CHAT CONTEXT:\n"
            for msg in history:
                role = "User" if msg.get("role") == "user" else "Assistant"
                # Handle old dictionary structures and standard SDK list structures
                content = ""
                if "parts" in msg:
                    content = " ".join([p.get("text", "") for p in msg["parts"]])
                else:
                    content = str(msg)
                history_str += f"- {role}: {content}\n\n"

        system_message = f"""You are a deep research assistant using DuckDuckGo web scraper results. 
Answer the user based on the provided search results and the previous context of this conversation. Cite sources using [1], [2] format inline.

{history_str}

{search_context}"""

        history_payload = []
        # Add system context
        history_payload.append({"role": "user", "parts": [{"text": system_message}]})
        # Note: we need a model response if we are simulating history. Or we just squash into current message to avoid role alternating errors
        # The new SDK is STRICT on role alternation.
        # To bypass, we will squash context into the user query.
        
        squashed_message = f"{system_message}\n\nUSER QUERY: {user_query}"
        
        try:
            response = await self._generate_with_fallback(contents=squashed_message)
            
            session.add_message("user", user_query)
            session.add_message("model", response.text)

            return {
                "response": response.text,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Chat generation error: {e}")
            return {"response": f"I encountered an API error: {str(e)}", "sources": []}

    def extract_sources(self, text: str) -> List[Dict[str, str]]:
        sources = []
        pattern = r'\[(\d+)\]\s*([^\n]*)\s*(?:URL:\s*(https?://[^\s\]]+))?'
        matches = re.findall(pattern, text)

        for match in matches:
            index_str, title, url = match
            if not url:
                url_search = re.search(rf'\[{index_str}\][^\n]*\n(https?://[^\s]+)', text)
                if url_search:
                    url = url_search.group(1)
            if not url: continue
            
            sources.append({
                "index": int(index_str), 
                "title": title.strip() if title else "Web Resource", 
                "url": url
            })
        return sources

    async def perform_research(
        self,
        user_query: str,
        depth: int,
        breadth: int,
        session_id: str,
        api_key: str = None
    ) -> Dict[str, Any]:
        
        self.configure_api(api_key)
        main_session = ChatSession(session_id)
        await main_session.load_history()

        research_log = {"query": user_query, "depth": depth, "breadth": breadth, "session_id": session_id, "levels": []}
        report_content = f"# Deep Research Report\n\n**Query:** {user_query}\n\n"
        all_sources = []

        try:
            # Level 1 - Unbounded Initial Assessment
            search_results = await self.search_web(user_query, 5) # Default 5 wide for level 1
            response_data = await self.chat_with_content(user_query, main_session, search_results)
            initial_sources = self.extract_sources(response_data["response"])
            if not initial_sources:
                 initial_sources = response_data.get("sources", []) # Fallback to original scraped links

            report_content += f"## Initial Findings\n{response_data['response']}\n\n"
            all_sources.extend(initial_sources)

            current_report = response_data["response"]
            
            # Sub-levels - LLM determines optimal breadth dynamically
            for d in range(2, depth + 1):
                follow_up_queries = await self.generate_search_queries(current_report)
                report_content += f"## Deeper Dive Level {d}\n\n"
                
                # To avoid complex async/genai client rate limit issues, we run sequentially instead of asyncio.gather()
                for i, q in enumerate(follow_up_queries, 1):
                    sub_session = ChatSession(f"{session_id}_d{d}_q{i}")
                    sub_search = await self.search_web(q, 2) # Limit to 2 per sub-query
                    sub_resp = await self.chat_with_content(q, sub_session, sub_search)
                    
                    sub_srcs = self.extract_sources(sub_resp["response"]) or sub_resp.get("sources", [])
                    report_content += f"### {q}\n{sub_resp['response']}\n\n"
                    all_sources.extend(sub_srcs)
                    current_report += f"\n{sub_resp['response']}"

            # Always run Planner-Reflector Gap-Fill using Flash Workers
            report_content += f"## Reflective Gap-Fill Analysis\n\n"
            critique_prompt = f"""Read this drafted research report. Identify exactly 3 critical pieces of missing, vague, or unsupported information. 
Return ONLY a valid JSON array of strings, where each string is a specific DuckDuckGo search query to find the missing data. Do not include markdown or explanations.
REPORT DRAFT:
{report_content[:40000]}"""
            try:
                critique_resp = await self._generate_with_fallback(contents=critique_prompt, role="worker")
                text = critique_resp.text.strip()
                text = re.sub(r'```[a-z]*\n(.*?)\n```', r'\1', text, flags=re.DOTALL).strip()
                gap_queries = json.loads(text)[:3]
                
                for i, q in enumerate(gap_queries, 1):
                    gap_session = ChatSession(f"{session_id}_gap_{i}")
                    gap_search = await self.search_web(q, 3)
                    gap_resp = await self.chat_with_content(q, gap_session, gap_search)
                    gap_srcs = self.extract_sources(gap_resp["response"]) or gap_resp.get("sources", [])
                    
                    report_content += f"### Gap-Fill Query: {q}\n{gap_resp['response']}\n\n"
                    all_sources.extend(gap_srcs)
            except Exception as e:
                logger.warning(f"Reflection loop failed or skipped: {e}")

            # Dedup sources
            unique_sources = {}
            for source in all_sources:
                if source["url"] not in unique_sources:
                    unique_sources[source["url"]] = source
            
            # Format final sources block
            if unique_sources:
                report_content += "## References\n"
                idx = 1
                for src in unique_sources.values():
                    report_content += f"[{idx}] {src['title']}\nURL: {src['url']}\n\n"
                    idx += 1

            # Final Orchestrator pass using Gemini 3.1 Pro
            polish_prompt = f"Rewrite this immense research report to be perfectly structured, professional, and comprehensive. Keep all markdown. Do not hallucinate.\n{report_content[:50000]}"
            final_resp = await self._generate_with_fallback(contents=polish_prompt, role="orchestrator")
            final_text = final_resp.text if final_resp else str(report_content)

            await self.save_research_to_db(session_id, user_query, final_text)

            return {
                "report": final_text,
                "research_log": research_log,
                "session_id": session_id
            }

        except Exception as e:
            logger.error(f"Research Error: {e}")
            return {"report": f"# Error\n\nResearch failed: {str(e)}\n\nPartial findings:\n{report_content}", "session_id": session_id, "error": str(e)}

    async def save_research_to_db(self, session_id: str, query: str, report: str):
        try:
            async with aiosqlite.connect(DATABASE) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO research_sessions (session_id, query, report, created_at) VALUES (?, ?, ?, ?)",
                    (session_id, query, report, datetime.now())
                )
                await db.commit()
        except Exception as e:
            logger.error(f"DB save error: {e}")

# --- FastAPI Routes ---
@app.get("/api/health")
async def health_check():
    return JSONResponse({"status": "healthy", "service": "Deep Research API Refactoring Complete"})

@app.post("/research")
async def research(research_request: ResearchRequest):
    user_query = research_request.query
    config_data = research_request.config
    session_id = config_data.get('sessionId', f"sess_{int(time.time())}")

    config = ResearchConfig(
        depth=config_data.get('depth', 2),
        breadth=config_data.get('breadth', 3),
        model_name=config_data.get('model', 'gemini-3.1-pro-preview'),
        citation_mode=config_data.get('citationMode', True)
    )

    research_tool = DeepResearchTool(config)

    async def generate():
        try:
            # Yield startup chunk
            yield f"data: {json.dumps({'status': 'Connecting to Deep Research Agent...'})}\n\n"
            
            result = await research_tool.perform_research(
                user_query=user_query,
                depth=config.depth,
                breadth=config.breadth,
                session_id=session_id,
                api_key=config_data.get('apiKey')
            )

            if "error" in result and result['report'].startswith("# Error"):
                 yield f"data: {json.dumps({'error': result['error']})}\n\n"
            else:
                 chunk_size = 500
                 for i in range(0, len(result['report']), chunk_size):
                     chunk = result['report'][i:i + chunk_size]
                     yield f"data: {json.dumps({'reportChunk': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type='text/event-stream')

@app.get("/previous_sessions")
async def get_previous_sessions(db: aiosqlite.Connection = Depends(get_db)):
    try:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT session_id, query, created_at FROM research_sessions ORDER BY created_at DESC") as cursor:
            sessions = []
            async for row in cursor:
                sessions.append({
                    "session_id": row["session_id"],
                    "query": row["query"],
                    "created_at": str(row["created_at"])
                })
            return JSONResponse(sessions)
    except Exception as e:
        logger.error(f"DB Error: {e}")
        return JSONResponse([]) # Ignore failures on vercel temp storage

@app.on_event("startup")
async def startup_event():
    await init_db()

if __name__ == "__main__":
     uvicorn.run(app, host="0.0.0.0", port=5000)