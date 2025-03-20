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
import google.generativeai as genai
from bs4 import BeautifulSoup
import aiosqlite
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound

from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class ResearchRequest(BaseModel):
    query: str
    config: Dict[str, Any]

class ConfigModel(BaseModel):
    depth: int = 2
    breadth: int = 3
    model: str = "gemini-2.0-flash-thinking-exp-01-21"
    citationMode: bool = True
    sessionId: str
    apiKey: str

# --- Configuration Class ---
@dataclass
class ResearchConfig:
    """Configuration settings for the research tool."""
    depth: int = 2              # Maximum recursion depth for research
    breadth: int = 3            # Number of search results to process per query
    max_content_length: int = 60000  # Maximum length of extracted content
    search_timeout: int = 30    # Timeout for web operations in seconds
    citation_mode: bool = True  # Enable/disable citation mode for reports
    model_name: str = "gemini-2.0-flash-thinking-exp-01-21"  # Default model to use

# --- FastAPI App Setup ---
app = FastAPI(title="Deep Research API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")

# --- Database Setup ---
DATABASE = "deep_research.db"

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

# --- Database Connection Helpers (for FastAPI) ---
async def get_db():
    """Returns a database connection."""
    async with aiosqlite.connect(DATABASE) as db:
        yield db

class ChatSession:
    """Manages conversation history with Gemini API, using aiosqlite."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history = []
        self.db_path = DATABASE
        
    async def load_history(self):
        """Load chat history from the database."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute("SELECT history FROM chat_sessions WHERE session_id=?", (self.session_id,)) as cursor:
                    row = await cursor.fetchone()
                    if row and row["history"]:
                        try:
                            self.history = json.loads(row["history"])
                        except json.JSONDecodeError:
                            logger.error(f"Failed to decode history for session {self.session_id}")
                            self.history = []
                    else:
                        self.history = []
        except Exception as e:
            logger.error(f"Failed to load chat history for session {self.session_id}: {e}")

    async def save_history(self):
        """Save chat history to the database."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Only save user and model messages, not the full prompt
                filtered_history = [
                    message for message in self.history
                    if message["role"] in ("user", "model")
                ]
                history_json = json.dumps(filtered_history)
                await db.execute("REPLACE INTO chat_sessions (session_id, history, updated_at) VALUES (?, ?, ?)",
                                (self.session_id, history_json, datetime.now()))
                await db.commit()
                logger.info(f"Saved chat history for session {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to save chat history for session {self.session_id}: {e}")

    def add_message(self, role: str, content: str):
        """Add a message to the chat history in Gemini-compatible format."""
        # Convert from role/content format to Gemini's format
        if role == "user":
            self.history.append({"role": "user", "parts": [{"text": content}]})
        elif role == "model":
            self.history.append({"role": "model", "parts": [{"text": content}]})
        elif role == "system":
            # For system messages, you may need to adjust based on Gemini's approach to system prompts
            self.history.append({"role": "user", "parts": [{"text": content}]})

        asyncio.create_task(self.save_history())

    def get_history(self):
        """Return the chat history."""
        return self.history

class DeepResearchTool:
    def __init__(self, config: ResearchConfig):
        """Initialize the research tool with configuration."""
        self.config = config
        self.model = None
        self.http_client = httpx.AsyncClient(timeout=config.search_timeout)

    def configure_api(self, api_key: Optional[str] = None):
        """Configure the Gemini API with the provided key or from environment."""
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("No API key provided and no environment variable found.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.config.model_name)
        logger.info(f"Gemini API configured with model {self.config.model_name}")
        return self.model

    async def search_web(self, query: str, num_results: int = None) -> List[Dict[str, str]]:
        """
        Search the web using Google's JSON Search API and return results.
        """
        if num_results is None:
            num_results = self.config.breadth

        results = []
        
        # Get the API key and CX from environment variables
        GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")  # Use the unified API key
        SEARCH_ENGINE_ID = os.environ.get("SEARCH_ENGINE_ID")  # Custom Search Engine ID
        
        if not GOOGLE_API_KEY or not SEARCH_ENGINE_ID:
            logger.error("Google Search API key or CX not configured")
            return [{"title": "Search configuration error", "url": "", "snippet": "Google Search API not properly configured."}]
        
        try:
            search_url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": GOOGLE_API_KEY,
                "cx": SEARCH_ENGINE_ID,
                "q": query,
                "num": min(num_results, 10)  # Google API limits to 10 results per request
            }
            
            response = await self.http_client.get(search_url, params=params)
            
            if response.status_code == 200:
                search_data = response.json()
                
                if "items" in search_data:
                    for item in search_data["items"]:
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("link", ""),
                            "snippet": item.get("snippet", "")
                        })
                else:
                    logger.warning(f"No search results found for query: {query}")
            else:
                logger.error(f"Google Search API error: {response.status_code} - {response.text}")
                return [{"title": "Search API error", "url": "", "snippet": f"Error {response.status_code} from Google Search API."}]
                
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return [{"title": "Search error", "url": "", "snippet": f"An error occurred during search: {str(e)}"}]
            
        # Always return something, even if empty
        if not results:
            return [{"title": "No results found", "url": "", "snippet": "No search results were found. Try modifying your query."}]
            
        return results

    async def extract_content(self, url: str) -> Optional[str]:
        """
        Extract content from a URL, with caching support.
        """
        # Check cache first
        cached_content = await self.get_cached_content(url)
        if cached_content:
            return cached_content

        # Extract based on URL type
        if "youtube.com" in url or "youtu.be" in url:
            content_data = await self.extract_from_youtube(url)
            if content_data and "error" not in content_data:
                 content = content_data["text"]
                 title = content_data["title"]
            else:
                content = None
                title = None
        else:
            content, title = await self.fetch_page_content(url)

        # Cache the content if successful
        if content:
            await self.cache_content(url, content, title or "")

        return content

    async def get_cached_content(self, url: str) -> Optional[str]:
        """Get content from cache if available and not expired."""
        try:
            async with aiosqlite.connect(DATABASE) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT content, updated_at FROM content_cache WHERE url = ?",
                    (url,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        # Check if cache is fresh (less than 24 hours old)
                        updated_at = datetime.fromisoformat(row["updated_at"])
                        if (datetime.now() - updated_at).total_seconds() < 86400:  # 24 hours
                            return row["content"]
            return None
        except Exception as e:
            logger.error(f"Error checking content cache: {e}")
            return None

    async def cache_content(self, url: str, content: str, title: str = ""):
        """Cache content for future use."""
        try:
            async with aiosqlite.connect(DATABASE) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO content_cache (url, content, title, updated_at) VALUES (?, ?, ?, ?)",
                    (url, content, title, datetime.now())
                )
                await db.commit()
        except Exception as e:
            logger.error(f"Error caching content: {e}")

    async def fetch_page_content(self, url: str) -> tuple[Optional[str], Optional[str]]:
        """
        Extract the main content from a webpage using httpx and BeautifulSoup.
        Returns (content, title) tuple.
        """
        try:
            response = await self.http_client.get(url, follow_redirects=True)
            response.raise_for_status()
            
            content = response.text
            soup = BeautifulSoup(content, 'html.parser')
            
            # Get page title
            title = soup.title.string if soup.title else "Untitled Page"
            
            # Remove script, style and irrelevant tags
            for element in soup(["script", "style", "nav", "footer", "header", "form"]):
                element.decompose()

            # Find main content area (adjust selector as needed)
            main_content = soup.find('article') or soup.find('main') or soup.body
            if not main_content:
                return "", title

            paragraphs = main_content.find_all(['p', 'h1', 'h2', 'h3', 'ul', 'ol'])
            text = '\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])

            # Truncate if too long
            if len(text) > self.config.max_content_length:
                text = text[:self.config.max_content_length] + "...[content truncated]"

            return text, title

        except httpx.TimeoutException:
            logger.error(f"Timeout fetching {url}")
            return None, None
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            return None, None

    async def extract_from_youtube(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract transcript and metadata from a YouTube video."""
        try:
            video_id = self._extract_video_id(url)
            if not video_id:
                return {"error": f"Failed to extract video ID from {url}"}

            # Try using YouTube Transcript API first
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                transcript_text = " ".join([entry['text'] for entry in transcript])
            except NoTranscriptFound:
                transcript_text = "[No transcript available]"

            # Get video title using YouTube API if available
            # For now, simplified to just extract from URL or use placeholder
            title = f"YouTube Video {video_id}"
            
            # Try to get more metadata using httpx
            try:
                response = await self.http_client.get(f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json")
                if response.status_code == 200:
                    data = response.json()
                    title = data.get("title", title)
            except Exception as e:
                logger.error(f"Error getting YouTube metadata: {e}")

            metadata = f"Title: {title}\nURL: {url}\n\n"
            return {"title": title, "text": f"{metadata}Transcript:\n{transcript_text}", "source": url, "type": "youtube"}

        except Exception as e:
            logger.error(f"Error extracting YouTube content from {url}: {e}")
            return {"error": f"Failed to extract content from YouTube video at {url}: {str(e)}"}

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from a URL."""
        parsed = urlparse(url)

        if parsed.netloc == 'youtu.be':
            return parsed.path[1:].split('?')[0]

        elif parsed.netloc in ('www.youtube.com', 'youtube.com'):
            if parsed.path == '/watch':
                query_params = parse_qs(parsed.query)
                return query_params.get('v', [None])[0]

            # Handle other YouTube URL formats
            path_segments = parsed.path.split('/')
            if len(path_segments) >= 3:
                if path_segments[1] in ['embed', 'v', 'shorts']:
                    return path_segments[2].split('?')[0]

        return None

    async def generate_search_queries(self, content: str, num_queries: int = None) -> List[str]:
        """Generate search queries based on content using Gemini."""
        if num_queries is None:
            num_queries = self.config.breadth

        if not self.model:
            raise ValueError("Gemini API not configured. Call configure_api() first.")

        try:
            prompt = f"""
            Based on the following information, generate {num_queries} specific search queries that would help gather more
            detailed information on the key topics mentioned. Format your response as a JSON array of strings.

            INFORMATION:
            {content}

            INSTRUCTIONS:
            - Generate {num_queries} search queries
            - Make queries specific and focused
            - Format as a JSON array of strings (e.g., ["query 1", "query 2"])
            - Do NOT include additional text, only the JSON
            """

            response = self.model.generate_content(prompt)
            response_text = response.text
            response_text = re.sub(r'```(?:json)?\n?(.*?)```', r'\1', response_text, flags=re.DOTALL)

            # Extract the JSON array
            try:
                # Try direct JSON parsing first
                queries = json.loads(response_text)
                if isinstance(queries, list):
                    return queries[:num_queries]

                # More robust JSON extraction using regex if needed
                json_match = re.search(r'\[[^\]]*\]', response_text)
                if json_match:
                    queries_json = json_match.group(0)
                    queries = json.loads(queries_json)
                    return queries[:num_queries]

                # Fallback to line splitting
                queries = [q.strip() for q in response_text.split('\n') if q.strip()]
                return queries[:num_queries]

            except json.JSONDecodeError:
                logger.warning(f"JSON decoding failed for: {response_text}")
                # Fallback: try splitting by lines if JSON parsing fails
                queries = [q.strip() for q in response_text.split('\n') if q.strip()]
                return queries[:num_queries]

        except Exception as e:
            logger.error(f"Error generating search queries: {e}")
            return []  # Return empty list on error

    async def chat_with_content(
        self,
        user_query: str,
        session: ChatSession,
        search_results: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Interact with Gemini, incorporating search results and chat history."""
        if not self.model:
            raise ValueError("Gemini API not configured. Call configure_api() first.")

        try:
            # Prepare search context if provided
            search_context = ""
            sources = []
            if search_results:
                search_context = "Search results:\n"
                for i, result in enumerate(search_results, 1):
                    search_context += f"[{i}] {result['title']}\nURL: {result['url']}\nSnippet: {result['snippet']}\n\n"
                    sources.append({
                        "index": i,
                        "title": result['title'],
                        "url": result['url']
                    })

            # Build the system message
            system_message = """
            **User query: {user_query}**

            {search_context}

            You are a deep research assistant. Your task is to provide comprehensive, accurate, and well-sourced answers.

            GUIDELINES:

            **General Interaction:**

            1.  Answer based on the provided sources when available.
            2.  If sources contradict, acknowledge different perspectives.
            3.  If information is unavailable in the sources, state that.
            4.  Organize complex answers with clear structure (headings, bullet points).
            5.  Focus on facts, cite sources.
            6.  Include publication dates when relevant.

            **Citations (if Citation Mode is enabled):**

            7.  Include citations in your response using the `[n]` format (e.g., `[1]`, `[2]`).
            8.  At the end, provide a "Sources:" section listing all cited sources with titles and URLs.

            **Response Structure and Formatting:**

            Your responses should be well-structured and easy to understand.  Follow these specific formatting rules:

            9.  Begin with a brief **executive summary** of key findings (2-3 sentences).
            10. Structure your response with clear **hierarchical headings** using Markdown syntax (`#` for main sections, `##` for subsections, etc.).
            11. Maintain a logical flow between sections, with smooth transitions.
            12. End with a concise **conclusion** summarizing the main insights.
            13. Use Markdown syntax to enhance readability:
                *   Use **# Headings** for main sections and **## Subheadings** for subsections
                *   Employ **bulleted lists** for related points:
                    *   Use `-` for unordered lists
                    *   Use `1.` for ordered lists or sequential steps
                *   Emphasize important concepts with **bold** and *italics*
                *   Use `inline code` for technical terms, commands, or variables
                *   Present code snippets in code blocks with language specification:

                    ```python
                    def example_function(parameter):
                        return parameter * 2
                    ```

                *   Use `>` blockquotes for definitions or important statements
                *   Employ `~~strikethrough~~` for outdated information or common misconceptions
                *   Insert horizontal rules (`---`) to separate major sections
                *   Use `[hyperlinks](URL)` for citing sources or additional reading
            14. For research involving mathematical concepts:
                *   Use `$inline math$` for equations within text (e.g., `$E = mc^2$`)
                *   Use `$$display math$$` for standalone equations:

                    $$\int_{a}^{b} f(x) dx = F(b) - F(a)$$
            15. Cite sources using numbered references `[1]`, `[2]`, etc. (aligns with item 7).
            16.  Include a "Sources" or "References" section at the end listing all cited sources (aligns with item 8).
            17. When presenting controversial findings, acknowledge differing viewpoints.

            **Clarity, Accuracy, and Depth:**

            18. Define specialized terminology.
            19. Provide specific examples to illustrate abstract concepts.
            20. Use analogies when appropriate to explain complex ideas.
            21. Quantify findings when possible (statistics, percentages, measurements).
            22. Acknowledge limitations and uncertainty in the research.
            23. Adjust the depth and comprehensiveness of your response according to the user-specified research level (this will be managed by the system, you don't need to explicitly ask for it).  Higher depth levels require more detail.
            24. Tailor the breadth (scope of topics) to the user-specified breadth (this is also managed by the system).

            **Contextual Information:**

            You may be provided with search results in the following format:

            ```
            Search results:
            [1] Title of Result 1
            URL: [url1]
            Snippet: [snippet1]

            [2] Title of Result 2
            URL: [url2]
            Snippet: [snippet2]

            ...
            ```

            Use these results, *along with your existing knowledge*, to answer the user's query.  Prioritize information from the provided search results, but supplement them with your knowledge as needed to create a complete and coherent response.  Always cite sources appropriately.  If search results are not provided, rely on your internal knowledge, but still indicate where information is derived from, if possible.
            Remember you will also be able to see previous messages in your history.
                """

            # Prepare gemini_history with system message and previous history only
            gemini_history = [{"role": "user", "parts": [{"text": system_message}]}]
            gemini_history.extend(session.get_history())

            # Start chat with previous history
            chat = self.model.start_chat(history=gemini_history)

            # Prepare the current message (include search context if available)
            if search_context:
                current_message = f"{search_context}\n\nUser query: {user_query}"
            else:
                current_message = user_query

            # Send the current message
            response = await chat.send_message_async(current_message)

            # Add the current message and response to session history
            session.add_message("user", current_message)
            session.add_message("model", response.text)

            return {
                "response": response.text,
                "sources": sources
            }

        except Exception as e:
            logger.error(f"Error in chat_with_content: {e}")
            error_response = f"I encountered an error: {str(e)}"
            session.add_message("model", error_response)
            return {"response": error_response, "sources": []}

    def extract_sources(self, text: str) -> List[Dict[str, str]]:
        """Extract sources from the generated text using regex."""
        sources = []
        # Updated regex to handle cases with or without titles
        pattern = r'\[(\d+)\]\s*([^\n]*)\s*(?:URL:\s*(https?://[^\s\]]+))?'
        matches = re.findall(pattern, text)

        for match in matches:
            index_str, title, url = match
            index = int(index_str)

            # If no explicit URL is found, try to find it nearby
            if not url:
                url_search = re.search(rf'\[{index}\][^\n]*\n(https?://[^\s]+)', text)
                if url_search:
                    url = url_search.group(1)

            # If still no URL, skip this source
            if not url:
                continue

            # Use a provided title or try to get it
            if title:
                sources.append({"index": index, "title": title.strip(), "url": url})
            else:
                sources.append({"index": index, "title": "Retrieved Content", "url": url})

        return sources

    async def perform_research(
        self,
        user_query: str,
        depth: int = None,
        breadth: int = None,
        session_id: str = None,
        api_key: str = None
    ) -> Dict[str, Any]:
        """Perform iterative deep research."""

        # Use config values if not specified
        if depth is None:
            depth = self.config.depth
        if breadth is None:
            breadth = self.config.breadth
        if not session_id:
            session_id = f"deep_research_{int(time.time())}"

        # Configure API key (passed from frontend)
        self.configure_api(api_key)

        # Check if Gemini API is configured
        if not self.model:
            raise ValueError("Gemini API not configured. Call configure_api() first.")

        main_session = ChatSession(session_id)
        await main_session.load_history() # load history

        research_log = {
            "query": user_query,
            "depth": depth,
            "breadth": breadth,
            "session_id": session_id,
            "levels": []
        }

        report_content = f"# Deep Research Report: {user_query}\n\n"
        all_sources = []

        try:
            # --- Level 1: Initial Research ---
            logger.info(f"Starting depth level 1: {user_query}")
            search_results = await self.search_web(user_query, breadth)
            response_data = await self.chat_with_content(user_query, main_session, search_results)
            initial_sources = self.extract_sources(response_data["response"])  # Use improved extraction

            research_log["levels"].append({
                "depth": 1,
                "query": user_query,
                "search_results": search_results,
                "response": response_data["response"],
                "sources": initial_sources  # Store extracted sources
            })
            report_content += f"## Initial Research\n\n{response_data['response']}\n\n"
            all_sources.extend(initial_sources) # Store extracted sources

            # --- Subsequent Depth Levels ---
            current_report = response_data["response"] # prepare for next level prompt
            for d in range(2, depth + 1):
                logger.info(f"Starting depth level {d}")
                follow_up_queries = await self.generate_search_queries(current_report, breadth)
                report_content += f"## Research Depth Level {d}\n\n"
                level_details = {
                    "depth": d,
                    "follow_up_queries": follow_up_queries,
                    "responses": []
                }

                for i, query in enumerate(follow_up_queries, 1):
                    logger.info(f"Researching question {i}/{len(follow_up_queries)} at depth {d}: {query}")
                    sub_session_id = f"{session_id}_depth{d}_question{i}"
                    sub_session = ChatSession(sub_session_id)  # Separate session for each question
                    await sub_session.load_history() # load history
                    sub_search_results = await self.search_web(query, breadth)
                    sub_response_data = await self.chat_with_content(query, sub_session, sub_search_results)
                    sub_sources = self.extract_sources(sub_response_data["response"]) # Extract the sources

                    level_details["responses"].append({
                        "query": query,
                        "search_results": sub_search_results,
                        "response": sub_response_data["response"],
                        "sources": sub_sources  # Store extracted sources
                    })
                    report_content += f"### Question: {query}\n\n{sub_response_data['response']}\n\n"
                    all_sources.extend(sub_sources) # Store extracted sources
                    current_report += f"\n\n{query}\n{sub_response_data['response']}" # update current report

                research_log["levels"].append(level_details)

            # --- Consolidated Sources ---
            unique_sources = {}  # Use a dict for efficient deduplication
            for source in all_sources:
                url = source["url"]
                if url not in unique_sources:
                    unique_sources[url] = source
            source_list = list(unique_sources.values())
            source_list.sort(key=lambda x: x["index"])  # Sort by original index

            report_content += "## Sources\n\n"
            for source in source_list:
                report_content += f"[{source['index']}] {source['title']}\n{source['url']}\n\n"

            # --- Final Report Generation (with Gemini) ---
            final_report_prompt = f"""
            Create a comprehensive, well-organized research report from the following content.
            Structure the report logically with headings and subheadings.
            Include ALL original citations and the Sources section. Maintain a consistent style.

            CONTENT:
            {report_content}

            GUIDELINES:
            1. Keep existing citations in the [n] format.
            2. Include all sources in the "Sources" section.
            3. Add a short executive summary at the beginning.
            4. Format the report in Markdown.
            """

            final_chat_session = ChatSession(f"{session_id}_final")  # Separate session
            await final_chat_session.load_history() # load history
            final_report_data = await self.chat_with_content(final_report_prompt, final_chat_session)

            # Save to database
            await self.save_research_to_db(session_id, user_query, final_report_data["response"])

            return {
                "report": final_report_data["response"],
                "research_log": research_log,
                "session_id": session_id
            }

        except Exception as e:
            logger.error(f"Error in perform_deep_research: {e}")
            error_report = f"# Error Report\n\nAn error occurred: {str(e)}\n\n"
            if 'current_report' in locals():  # Check if current_report exists
                error_report += f"Partial research:\n\n{current_report}"
            return {"report": error_report, "research_log": research_log, "error": str(e), "session_id": session_id}

    async def save_research_to_db(self, session_id: str, query: str, report: str):
        """Save research report to database."""
        try:
            async with aiosqlite.connect(DATABASE) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO research_sessions (session_id, query, report, created_at) VALUES (?, ?, ?, ?)",
                    (session_id, query, report, datetime.now())
                )
                await db.commit()
                logger.info(f"Saved research report for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to save research report: {e}")

    async def get_saved_research(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a saved research report by session ID."""
        try:
            async with aiosqlite.connect(DATABASE) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT query, report, created_at FROM research_sessions WHERE session_id = ?",
                    (session_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        return {
                            "session_id": session_id,
                            "query": row["query"],
                            "report": row["report"],
                            "created_at": row["created_at"]
                        }
            return None
        except Exception as e:
            logger.error(f"Error retrieving research report: {e}")
            return None

# --- FastAPI Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

from fastapi.responses import HTMLResponse

@app.post("/research")
async def research(research_request: ResearchRequest):
    """Handle research requests from the frontend."""
    user_query = research_request.query
    config_data = research_request.config
    session_id = config_data['sessionId']

    # Create a ResearchConfig object from the frontend data
    config = ResearchConfig(
        depth=config_data['depth'],
        breadth=config_data['breadth'],
        model_name=config_data['model'],
        citation_mode=config_data['citationMode']
    )

    research_tool = DeepResearchTool(config)

    async def generate():
        """Streams the research response asynchronously."""
        try:
            # Perform the research
            result = await research_tool.perform_research(
                user_query=user_query,
                depth=config.depth,
                breadth=config.breadth,
                session_id=session_id,
                api_key=config_data['apiKey']
            )

            # Stream the report (or error)
            if "error" in result:
                yield f"data: {json.dumps({'error': result['error']})}\n\n"
            else:
                # Chunk the report for streaming
                chunk_size = 500  # Adjust as needed
                for i in range(0, len(result['report']), chunk_size):
                    chunk = result['report'][i:i + chunk_size]
                    yield f"data: {json.dumps({'reportChunk': chunk})}\n\n"

        except Exception as e:
            logger.exception("Error during research:")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type='text/event-stream')

@app.get("/previous_sessions")
async def get_previous_sessions(db: aiosqlite.Connection = Depends(get_db)):
    """Retrieves a list of previous research sessions."""
    try:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT session_id, query, created_at FROM research_sessions ORDER BY created_at DESC") as cursor:
            sessions = []
            async for row in cursor:
                sessions.append({
                    "session_id": row["session_id"],
                    "query": row["query"],
                    "created_at": row["created_at"] if isinstance(row["created_at"], str) else row["created_at"].isoformat()
                })
            return JSONResponse(sessions)
    except Exception as e:
        logger.error(f"Error retrieving previous sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve previous sessions")

@app.get("/load_session/{session_id}")
async def load_session(session_id: str):
    """Loads a specific research session."""
    try:
        research_tool = DeepResearchTool(ResearchConfig()) # Dummy config
        result = await research_tool.get_saved_research(session_id)
        if result:
             return JSONResponse(result)
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error loading session: {e}")
        raise HTTPException(status_code=500, detail="Failed to load session")
    

@app.on_event("startup")
async def startup_event():
    """Initialize the database on application startup."""
    await init_db()

if __name__ == "__main__":
    # Run the app with uvicorn for proper async support
     uvicorn.run(app, host="0.0.0.0", port=5000)

# Run using: uvicorn app:app --reload