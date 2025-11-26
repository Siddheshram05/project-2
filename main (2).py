import os
import base64
import json
import asyncio
import re
import logging
from logging.handlers import RotatingFileHandler
import httpx
import io
from typing import Any, List, Dict, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from playwright.async_api import async_playwright
from openai import OpenAI
from dotenv import load_dotenv
import pypdf
import pandas as pd 

# --- Configuration ---
load_dotenv()


if os.path.exists("/data"):
    LOG_DIR = "/data/logs"
else:
    LOG_DIR = "/app/logs"

os.makedirs(LOG_DIR, exist_ok=True)

# --- Dual Logging Setup ---
def setup_logging():
    """Configure dual logging: detailed system log + human-readable quiz summary"""
    
    # 1. SYSTEM DEBUG LOG
    system_logger = logging.getLogger('system')
    system_logger.setLevel(logging.DEBUG)
    system_logger.propagate = False
    
    system_handler = RotatingFileHandler(
        f'{LOG_DIR}/system_debug.log',
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    system_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    system_handler.setFormatter(system_formatter)
    system_logger.addHandler(system_handler)
    
    # Console handler for Render dashboard
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(system_formatter)
    system_logger.addHandler(console_handler)
    
    # 2. QUIZ SUMMARY LOG
    quiz_logger = logging.getLogger('quiz_summary')
    quiz_logger.setLevel(logging.INFO)
    quiz_logger.propagate = False
    
    quiz_handler = RotatingFileHandler(
        f'{LOG_DIR}/quiz_summary.log',
        maxBytes=5*1024*1024,
        backupCount=3,
        encoding='utf-8'
    )
    quiz_formatter = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    quiz_handler.setFormatter(quiz_formatter)
    quiz_logger.addHandler(quiz_handler)
    
    # Also to console
    quiz_console = logging.StreamHandler()
    quiz_console.setFormatter(quiz_formatter)
    quiz_logger.addHandler(quiz_console)
    
    return system_logger, quiz_logger

# Initialize loggers
logger, quiz_log = setup_logging()

EMAIL = "23f3000704@study.iitm.ac.in"
SECRET = "Ksid2005"

# AIPipe API setup
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")

if not AIPIPE_TOKEN:
    logger.warning("[WARNING] AIPIPE_TOKEN not set! Set it in Render environment variables")

client = OpenAI(
    api_key=AIPIPE_TOKEN,
    base_url="https://aipipe.org/openrouter/v1"
)

# --- Helper Functions ---
def extract_text_from_pdf_bytes(content: bytes) -> str:
    """Extract text from PDF bytes"""
    try:
        pdf_file = io.BytesIO(content)
        reader = pypdf.PdfReader(pdf_file)
        text_parts = []
        for i, page in enumerate(reader.pages):
            text_parts.append(f"\n--- Page {i+1} ---\n")
            text_parts.append(page.extract_text())
        return ''.join(text_parts)
    except Exception as e:
        logger.error(f"PDF extraction error: {e}", exc_info=True)
        return f"Error reading PDF: {str(e)}"

def log_quiz_summary(quiz_url: str, question: str, data_urls: List[str], answer: Any, result: Dict):
    """Log human-readable quiz summary"""
    separator = "=" * 80
    quiz_log.info(f"\n{separator}")
    quiz_log.info(f"QUIZ URL: {quiz_url}")
    quiz_log.info(f"QUESTION: {question}")
    
    if data_urls:
        quiz_log.info(f"DATA SOURCES ({len(data_urls)}):")
        for url in data_urls:
            quiz_log.info(f"  - {url}")
    else:
        quiz_log.info("DATA SOURCES: None")
    
    quiz_log.info(f"ANSWER: {answer}")
    
    if result.get("correct"):
        quiz_log.info("RESULT: [CORRECT]")
        if result.get("url"):
            quiz_log.info(f"NEXT QUIZ: {result['url']}")
    else:
        quiz_log.info(f"RESULT: [INCORRECT] - {result.get('reason', 'Unknown')}")
    
    quiz_log.info(f"{separator}\n")

# --- Startup Event ---
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("="*60)
    logger.info("Starting Quiz Solver on Render...")
    logger.info(f"Email: {EMAIL}")
    logger.info(f"AIPipe Token Set: {bool(AIPIPE_TOKEN)}")
    logger.info(f"Log Directory: {LOG_DIR}")
    logger.info(f"System log: {LOG_DIR}/system_debug.log")
    logger.info(f"Quiz summary log: {LOG_DIR}/quiz_summary.log")
    
    # Test LLM connection
    if AIPIPE_TOKEN:
        try:
            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=20
            )
            logger.info(f"[OK] LLM Connection OK: {response.choices[0].message.content}")
        except Exception as e:
            logger.error(f"[FAIL] LLM Connection FAILED: {e}", exc_info=True)
    logger.info("="*60)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Quiz Solver...")

app = FastAPI(lifespan=lifespan)

# --- API Routes ---

@app.post("/quiz")
async def handle_quiz(request: Request):
    """Main endpoint to receive quiz tasks"""
    try:
        data = await request.json()
        logger.debug(f"Received request: {json.dumps(data, indent=2)}")
    except Exception as e:
        logger.error(f"Invalid JSON: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    if data.get("secret") != SECRET:
        logger.warning(f"Invalid secret attempt: {data.get('secret')}")
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    if data.get("email") != EMAIL:
        logger.warning(f"Invalid email attempt: {data.get('email')}")
        raise HTTPException(status_code=403, detail="Invalid email")
    
    quiz_url = data.get("url")
    if not quiz_url:
        logger.error("Missing URL in request")
        raise HTTPException(status_code=400, detail="Missing URL")
    
    logger.info(f"[OK] Received valid quiz request for: {quiz_url}")
    
    asyncio.create_task(solve_quiz_chain(quiz_url))
    
    return JSONResponse({"status": "processing", "url": quiz_url})

async def solve_quiz_chain(initial_url: str):
    """Solve a chain of quizzes with 3-minute timeout"""
    current_url = initial_url
    max_iterations = 15
    start_time = asyncio.get_event_loop().time()
    
    logger.info(f"Starting quiz chain from: {initial_url}")
    
    for iteration in range(max_iterations):
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > 170:
            logger.error(f"[TIMEOUT] Timeout approaching! Stopping at {elapsed:.1f}s")
            break
            
        logger.info(f"\n{'='*60}")
        logger.info(f"Iteration {iteration+1}: {current_url}")
        logger.info(f"Time elapsed: {elapsed:.1f}s")
        logger.info(f"{'='*60}\n")
        
        try:
            quiz_content = await fetch_quiz_page(current_url)
            answer, submit_url, question, data_urls = await solve_quiz_with_llm(quiz_content, current_url)
            
            if not submit_url:
                logger.error("[FAIL] No submit URL found")
                break

            result = await submit_answer(submit_url, current_url, answer)
            log_quiz_summary(current_url, question, data_urls, answer, result)
            
            if result.get("correct"):
                logger.info("[OK] CORRECT ANSWER!")
                next_url = result.get("url")
                if next_url:
                    current_url = next_url
                else:
                    logger.info("[SUCCESS] Quiz chain completed successfully!")
                    break
            else:
                reason = result.get('reason', 'Unknown')
                logger.warning(f"[FAIL] WRONG ANSWER: {reason}")
                
                next_url = result.get("url")
                if next_url and next_url != current_url:
                    logger.info(f"Moving to next quiz: {next_url}")
                    current_url = next_url
                else:
                    logger.warning("No new URL provided. Stopping.")
                    break
                    
        except Exception as e:
            logger.error(f"[ERROR] Error in iteration {iteration+1}: {e}", exc_info=True)
            break
    
    total_time = asyncio.get_event_loop().time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Quiz chain ended after {iteration+1} iterations in {total_time:.1f}s")
    logger.info(f"{'='*60}\n")

async def fetch_quiz_page(url: str) -> str:
    """Fetch and render JavaScript-enabled page"""
    logger.info(f"[FETCH] Fetching page: {url}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        try:
            logger.debug(f"Navigating to {url}")
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(2000)
            
            html_content = await page.content()
            text_content = await page.evaluate("document.body.innerText")
            
            await browser.close()
            
            logger.info(f"[OK] Page fetched successfully ({len(html_content)} chars)")
            logger.debug(f"Text content preview: {text_content[:200]}...")
            return f"HTML Source:\n{html_content}\n\nVisible Text:\n{text_content}"
            
        except Exception as e:
            await browser.close()
            logger.error(f"Playwright error: {e}", exc_info=True)
            raise Exception(f"Playwright error: {e}")

async def solve_quiz_with_llm(quiz_content: str, quiz_url: str) -> tuple:
    """Two-step solving: 1) Plan, 2) Execute
    Returns: (answer, submit_url, question, data_urls)"""
    
    planning_prompt = f"""You are solving a data analysis quiz.

PAGE CONTENT:
{quiz_content}

TASK: Analyze this page and extract:
1. The QUESTION being asked
2. The SUBMIT URL (where to POST the answer)
3. Any DATA URLs that need to be downloaded (CSV, PDF, JSON, etc.)
4. The expected ANSWER TYPE (number, string, boolean, json)

Current page URL: {quiz_url}

Return ONLY valid JSON:
{{
    "question": "the question text",
    "submit_url": "exact URL to submit answer",
    "data_urls": ["url1", "url2"],
    "answer_type": "number|string|boolean|json",
    "reasoning": "brief explanation"
}}"""
    
    try:
        logger.info("[PLAN] Planning with LLM...")
        logger.debug(f"Planning prompt length: {len(planning_prompt)} chars")
        
        response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": "You extract structured information from quiz pages. Always return valid JSON."},
                {"role": "user", "content": planning_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        plan = json.loads(response.choices[0].message.content)
        logger.info(f"[PLAN] Plan: {json.dumps(plan, indent=2)}")
        
    except Exception as e:
        logger.error(f"Planning failed: {e}", exc_info=True)
        return None, None, "Error: Planning failed", []

    context_data = ""
    data_urls = plan.get("data_urls", [])
    if data_urls:
        logger.info(f"[DATA] Downloading {len(data_urls)} data sources...")
        context_data = await download_data(data_urls)

    solve_prompt = f"""Question: {plan['question']}

Data Context:
{context_data if context_data else "No external data needed"}

Calculate the answer.
- If answer_type is 'number': return just the number (e.g., 42 or 3.14)
- If answer_type is 'string': return just the string
- If answer_type is 'boolean': return true or false
- If answer_type is 'json': return valid JSON

Respond with ONLY the answer value, no explanation."""
    
    try:
        logger.info("[SOLVE] Calculating answer...")
        logger.debug(f"Solve prompt length: {len(solve_prompt)} chars")
        
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise data analyst. Provide only the answer."},
                {"role": "user", "content": solve_prompt}
            ],
            temperature=0
        )
        
        raw_answer = response.choices[0].message.content.strip()
        logger.info(f"Raw answer: {raw_answer}")
        
        answer = raw_answer
        answer = re.sub(r'```.*?```', '', answer, flags=re.DOTALL).strip()
        
        if plan['answer_type'] == 'number':
            nums = re.findall(r'-?\d+\.?\d*', answer)
            if nums:
                answer = float(nums[0]) if '.' in nums[0] else int(nums[0])
        elif plan['answer_type'] == 'boolean':
            answer = answer.lower() in ['true', 'yes', '1']
        elif plan['answer_type'] == 'json':
            try:
                answer = json.loads(answer)
            except:
                json_match = re.search(r'\{.*\}', answer, re.DOTALL)
                if json_match:
                    answer = json.loads(json_match.group())
        
        logger.info(f"[OK] Final answer: {answer} (type: {type(answer).__name__})")
        return answer, plan.get("submit_url"), plan.get("question", "Unknown"), data_urls
        
    except Exception as e:
        logger.error(f"Solving failed: {e}", exc_info=True)
        return None, None, plan.get("question", "Unknown"), data_urls

async def download_data(urls: List[str]) -> str:
    """Download and extract text from various file formats"""
    combined = []
    
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        for url in urls:
            try:
                logger.info(f"  [DOWNLOAD] Downloading: {url}")
                resp = await http_client.get(url)
                content_type = resp.headers.get("content-type", "").lower()
                logger.debug(f"  Content-Type: {content_type}, Size: {len(resp.content)} bytes")
                
                if "pdf" in content_type or url.endswith(".pdf"):
                    text = extract_text_from_pdf_bytes(resp.content)
                    combined.append(f"\n--- PDF: {url} ---\n{text}")
                    
                elif "csv" in content_type or url.endswith(".csv"):
                    combined.append(f"\n--- CSV: {url} ---\n{resp.text}")
                    
                elif "json" in content_type or url.endswith(".json"):
                    combined.append(f"\n--- JSON: {url} ---\n{resp.text}")
                    
                elif "excel" in content_type or url.endswith((".xlsx", ".xls")):
                    try:
                        df = pd.read_excel(io.BytesIO(resp.content))
                        combined.append(f"\n--- Excel: {url} ---\n{df.to_string()}")
                        logger.info(f"  [OK] Parsed Excel: {df.shape[0]} rows, {df.shape[1]} columns")
                    except Exception as e:
                        logger.error(f"  [FAIL] Excel parsing error: {e}", exc_info=True)
                        combined.append(f"\n--- Excel Error: {url} ---\n{str(e)}")
                    
                else:
                    combined.append(f"\n--- Data: {url} ---\n{resp.text[:5000]}")
                    
                logger.info(f"  [OK] Downloaded successfully")
                
            except Exception as e:
                logger.error(f"  [FAIL] Error downloading {url}: {e}", exc_info=True)
                combined.append(f"\n--- Error: {url} ---\n{e}")
    
    return '\n'.join(combined)

async def submit_answer(submit_url: str, original_url: str, answer: Any) -> Dict:
    """Submit answer to quiz endpoint"""
    payload = {
        "email": EMAIL,
        "secret": SECRET,
        "url": original_url,
        "answer": answer
    }
    
    logger.info(f"[SUBMIT] Submitting to: {submit_url}")
    logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
    
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        try:
            resp = await http_client.post(submit_url, json=payload)
            result = resp.json()
            logger.info(f"Response status: {resp.status_code}")
            logger.debug(f"Response: {json.dumps(result, indent=2)}")
            return result
        except Exception as e:
            logger.error(f"Submission error: {e}", exc_info=True)
            return {"correct": False, "reason": str(e)}

@app.get("/")
def home():
    return {
        "status": "running",
        "project": "LLM Analysis Quiz Solver",
        "email": EMAIL,
        "platform": "Render",
        "endpoints": {
            "quiz": "/quiz (POST)",
            "health": "/health (GET)"
        },
        "logs": {
            "directory": LOG_DIR,
            "system": f"{LOG_DIR}/system_debug.log",
            "summary": f"{LOG_DIR}/quiz_summary.log"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "api_key_set": bool(AIPIPE_TOKEN),
        "log_dir": LOG_DIR,
        "logs_exist": os.path.exists(f"{LOG_DIR}/system_debug.log")
    }

# For local testing
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
