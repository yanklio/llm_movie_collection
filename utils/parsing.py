import json
import re
from typing import Any, Dict, Optional

def extract_json_from_response(content: str) -> Dict[str, Any]:
    """
    Robustly extract JSON from an LLM response string.
    
    Handles:
    - <think> blocks (removes them)
    - Markdown code blocks (```json ... ```)
    - Raw JSON strings
    - Dirty JSON (text before/after)
    
    Returns empty dict if parsing fails.
    """
    try:
        if not content:
            return {}
            
        # 1. Remove <think> blocks (common in reasoning models)
        if "<think>" in content:
            # content = content.split("</think>")[-1].strip()
            # More robust regex replace in case of multiple or malformed tags
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        # 2. Extract from Markdown code blocks
        json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_block_match:
            content = json_block_match.group(1)
        else:
            # 3. If no code blocks, try to find the first { and last }
            # This handles cases where LLM just outputs JSON with some text around it
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                content = content[start : end + 1]

        # 4. Parse
        return json.loads(content)
        
    except Exception as e:
        # Caller should handle logging if needed
        raise ValueError(f"Failed to parse JSON: {e}") from e
