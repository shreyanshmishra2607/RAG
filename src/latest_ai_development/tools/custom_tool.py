from crewai.tools import BaseTool
from typing import Type, Dict, Any
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup


class ScrapeInput(BaseModel):
    """Input schema for ScrapeWebsiteCustomTool."""
    url: str = Field(..., description="The URL of the website to scrape.")


class ScrapeWebsiteCustomTool(BaseTool):
    name: str = "Scrape Website Tool"
    description: str = "Scrapes and extracts content from a given website URL."
    args_schema: Type[BaseModel] = ScrapeInput

    def _run(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            return text[:5000]  # Limit output for LLM context safety
        except Exception as e:
            return f"Error scraping {url}: {str(e)}"
