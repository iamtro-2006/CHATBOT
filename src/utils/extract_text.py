def extract_clean_text(content) -> str:
    """Extract clean text from Gemini response."""
    if not content:
        return ""
    
    if isinstance(content, str):
        return content.strip()
    
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and 'text' in item:
                return item['text'].strip()
            elif isinstance(item, str):
                return item.strip()
    
    return str(content).strip() if content else ""