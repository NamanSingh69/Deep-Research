import asyncio
import httpx
from bs4 import BeautifulSoup
import urllib.parse

async def search_ddg(query, num_results=3):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    url = "https://html.duckduckgo.com/html/"
    async with httpx.AsyncClient() as client:
        req = await client.post(url, data={"q": query}, headers=headers)
        soup = BeautifulSoup(req.text, 'html.parser')
        results = []
        for result in soup.find_all('div', class_='result'):
            title_elem = result.find('a', class_='result__url')
            snippet_elem = result.find('a', class_='result__snippet')
            if title_elem and snippet_elem:
                title = title_elem.text.strip()
                link = title_elem.get('href')
                if link and link.startswith('//duckduckgo.com/l/?'):
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
                    link = parsed.get('uddg', [link])[0]
                snippet = snippet_elem.text.strip()
                results.append({"title": title, "url": link, "snippet": snippet})
                if len(results) >= num_results:
                    break
        print(results)

asyncio.run(search_ddg('Eiffel Tower'))
