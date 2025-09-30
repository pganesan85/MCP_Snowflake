from typing import Any, Dict, Tuple, List
import httpx
from mcp.server.fastmcp import FastMCP
import os
import json
import uuid
from dotenv import load_dotenv, find_dotenv
from tabulate import tabulate
import asyncio

load_dotenv(find_dotenv())

# Initialize FastMCP server
mcp = FastMCP("cortex_agent")

# Constants
SEMANTIC_MODEL_FILE = os.getenv("SEMANTIC_MODEL_FILE")
CORTEX_SEARCH_SERVICE = os.getenv("CORTEX_SEARCH_SERVICE")
SNOWFLAKE_ACCOUNT_URL = os.getenv("SNOWFLAKE_ACCOUNT_URL")
SNOWFLAKE_PAT = os.getenv("SNOWFLAKE_PAT")

if not SNOWFLAKE_PAT:
    raise RuntimeError("Set SNOWFLAKE_PAT environment variable")
if not SNOWFLAKE_ACCOUNT_URL:
    raise RuntimeError("Set SNOWFLAKE_ACCOUNT_URL environment variable")

API_HEADERS = {
    "Authorization": f"Bearer {SNOWFLAKE_PAT}",
    "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
    "Content-Type": "application/json",
}

# --- SSE parser ---
async def process_sse_response(resp: httpx.Response) -> Tuple[str, str, List[Dict]]:
    text, sql, citations = "", "", []

    async for raw_line in resp.aiter_lines():
        if not raw_line:
            continue
        raw_line = raw_line.strip()
        if not raw_line.startswith("data:"):
            continue

        payload = raw_line[len("data:"):].strip()
        if payload in ("", "[DONE]"):
            continue

        try:
            evt = json.loads(payload)
        except json.JSONDecodeError:
            continue

        # Safely extract delta
        delta = None
        if isinstance(evt, dict):
            if "delta" in evt and isinstance(evt["delta"], dict):
                delta = evt["delta"]
            elif isinstance(evt.get("data"), dict) and isinstance(evt["data"].get("delta"), dict):
                delta = evt["data"]["delta"]

        if not isinstance(delta, dict):
            continue

        content_items = delta.get("content", [])
        if not isinstance(content_items, list):
            continue

        for item in content_items:
            if not isinstance(item, dict):
                continue

            t = item.get("type")
            if t == "text":
                text += item.get("text", "")
            elif t == "tool_results":
                tool_results = item.get("tool_results", {})
                if not isinstance(tool_results, dict):
                    continue

                for result in tool_results.get("content", []) or []:
                    if not isinstance(result, dict):
                        continue

                    if result.get("type") == "json":
                        j = result.get("json", {})
                        if not isinstance(j, dict):
                            continue

                        text += j.get("text", "")

                        # capture SQL if present
                        if isinstance(j.get("sql"), str):
                            sql = j["sql"]

                        # capture any citations
                        search_results = j.get("searchResults", [])
                        if isinstance(search_results, list):
                            for s in search_results:
                                if isinstance(s, dict):
                                    citations.append({
                                        "source_id": s.get("source_id"),
                                        "doc_id": s.get("doc_id"),
                                    })
    return text, sql, citations

# --- Execute SQL in Snowflake ---
async def execute_sql(sql: str) -> Dict[str, Any]:
    request_id = str(uuid.uuid4())
    sql_api_url = f"{SNOWFLAKE_ACCOUNT_URL}/api/v2/statements"
    sql_payload = {
        "statement": sql.replace(";", ""),
        "timeout": 60
    }
    async with httpx.AsyncClient() as client:
        sql_response = await client.post(
            sql_api_url,
            json=sql_payload,
            headers=API_HEADERS,
            params={"requestId": request_id}
        )
        if sql_response.status_code == 200:
            return sql_response.json()
        else:
            return {"error": f"SQL API error: {sql_response.text}"}

# --- MCP tool ---
@mcp.tool()
async def run_cortex_agents(query: str) -> Dict[str, Any]:
    payload = {
        "model": "claude-3-5-sonnet",
        "response_instruction": "You are a helpful AI assistant.",
        "experimental": {},
        "tools": [
            {"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "Analyst1"}},
            {"tool_spec": {"type": "cortex_search", "name": "Search1"}},
            {"tool_spec": {"type": "sql_exec", "name": "sql_execution_tool"}},
        ],
        "tool_resources": {
            "Analyst1": {"semantic_model_file": SEMANTIC_MODEL_FILE},
            "Search1": {"name": CORTEX_SEARCH_SERVICE},
        },
        "tool_choice": {"type": "auto"},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": query}]}
        ],
    }

    request_id = str(uuid.uuid4())
    url = f"{SNOWFLAKE_ACCOUNT_URL}/api/v2/cortex/agent:run"
    headers = {**API_HEADERS, "Accept": "application/json"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, json=payload, headers=headers, params={"requestId": request_id}) as resp:
            resp.raise_for_status()
            text, sql, citations = await process_sse_response(resp)

    # --- Execute SQL if present ---
    results_table = None
    results_table_str = None
    if sql:
        try:
            results = await execute_sql(sql)
            data_rows = results.get("data", [])
            columns = [col["name"] for col in results.get("resultSetMetaData", {}).get("rowType", [])]
            results_table = [dict(zip(columns, row)) for row in data_rows]
            if results_table:
                results_table_str = tabulate(results_table, headers="keys", tablefmt="pretty")
        except Exception as e:
            results_table = [{"error": f"SQL execution failed: {e}"}]
            results_table_str = f"SQL execution failed: {e}"

    return {
        "text": text if text else "No text response generated.",
        "sql": sql if sql else "NO_SQL_GENERATED",
        "results": results_table,          # structured JSON
        "results_table": results_table_str,  # pretty table string
        "citations": citations if citations else []
    }

if __name__ == "__main__":
    mcp.run(transport='stdio')
    #host = os.getenv("MCP_HOST", "0.0.0.0")
    #port = int(os.getenv("MCP_PORT", "8000"))
    #os.environ["MCP_HOST"] = host
    #os.environ["MCP_PORT"] = str(port)
    #mcp.run(transport="http")

