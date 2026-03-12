import asyncio
import os
from app import DeepResearchTool, ResearchConfig

async def test_fallback():
    print("Testing gemini generation fallback...")
    
    # Try with a valid/invalid model using config default
    config = ResearchConfig()
    tool = DeepResearchTool(config)
    tool.configure_api() # Relies on the user's GEMINI_API_KEY env or we provide a dummy
    
    try:
        resp = await tool._generate_with_fallback("Write a 1 sentence summary of the moon.")
        print("Success!")
        print(f"Response: {resp.text}")
    except Exception as e:
        print(f"Failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(test_fallback())
