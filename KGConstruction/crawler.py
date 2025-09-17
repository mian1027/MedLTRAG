import requests
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import quote

# Configuration
SEARCH_QUERY = ""
BASE_URL = "https://pubmed.ncbi.nlm.nih.gov/"

# Dynamic User-Agent pool (modern browsers and mobile devices)
USER_AGENTS = [
    # Chrome
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.179 Safari/537.36",

    # Firefox
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0",

    # Safari
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5.1 Safari/605.1.15",

    # Edge
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",

    # Mobile devices
    "Mozilla/5.0 (iPhone14,3; U; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/602.1.50 (KHTML, like Gecko) Version/15.0 Mobile/19A346 Safari/602.1",
    "Mozilla/5.0 (Linux; Android 13; SM-S908B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36"
]

# Generate dynamic headers
def get_random_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": random.choice([
            "https://www.google.com/",
            "https://www.bing.com/",
            "https://www.baidu.com/",
            "https://www.yahoo.com/"
        ]),
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": str(random.randint(0, 1)),  # Random Do Not Track
    }

# Custom Session with dynamic headers and request interval
class SmartSession(requests.Session):
    def __init__(self):
        super().__init__()
        self.headers.update(get_random_headers())
        self.last_request_time = 0
        self.min_interval = random.uniform(1.5, 4.0)  # Dynamic request interval

    # Request with automatic delay and header rotation
    def smart_get(self, url):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed + random.uniform(0.2, 0.8)
            time.sleep(sleep_time)

        response = super().get(url, timeout=15)
        self.last_request_time = time.time()

        # Update headers if 403 Forbidden
        if response.status_code == 403:
            self.headers.update(get_random_headers())
            print("403 detected, automatically rotating headers")

        return response

# Generate full URL with pagination and year filter
def generate_search_url(page,YEAR_FILTER):
    encoded_query = quote(SEARCH_QUERY)
    return f"{BASE_URL}?term={encoded_query}&filter={YEAR_FILTER}&page={page}"

# Extract PMIDs with abstracts for a given year
def extract_pmid_with_abstracts(year):
    session = SmartSession()
    pmids = []
    max_retry = 10
    page = 1
    YEAR_FILTER = f"years.{year}-{year}"
    while True:
        for retry in range(max_retry):
            try:
                url = generate_search_url(page,YEAR_FILTER)
                response = session.smart_get(url)
                response.raise_for_status()

                # Check verification page
                if "bot check" in response.text.lower():
                    raise Exception("Triggered anti-bot verification")

                # Parse results
                soup = BeautifulSoup(response.text, 'html.parser')
                articles = soup.select('article.full-docsum')

                # Termination condition
                if not articles or "no results were found" in response.text.lower():
                    print(f"No results on page {page} , stopping crawl")
                    return pmids

                # Extract PMID
                page_pmids = []
                for article in articles:
                    pmid_tag = article.select_one('span.docsum-pmid')
                    abstract_tag = article.select_one('div.docsum-snippet')

                    if pmid_tag and abstract_tag:
                        pmid = pmid_tag.text.split(":")[-1].strip()
                        if pmid != None:
                            page_pmids.append(pmid)

                if page_pmids:
                    pmids.extend(page_pmids)
                    print(f"Page {page}: collected {len(pmids)} PMIDs so far")
                else:
                    print(f"Page {page} has no valid data, possibly reached the end")

                page_success = True

                page += 1
            except Exception as e:
                print(f"Request failed ({retry + 1}/{max_retry}): {str(e)}")
                if retry < max_retry - 1:
                    # After failure, rotate headers and proxies
                    session.headers.update(get_random_headers())
                    time.sleep(2 ** (retry + 1))  # Exponential backoff
                else:
                    print(f"Failed {max_retry} times consecutively, stopping crawl")
                    return pmids

            if page_success:
                time.sleep(random.uniform(1, 3))

    print(f"Reached maximum page limit {max_pages}，stopping crawl")
    return pmids

def save_PMID(year: str):
    result = extract_pmid_with_abstracts(year)
    result = list(set(result))
    print("\nFinal result:")
    print(f"Found {len(result)} PMIDs with abstracts")
    with open(f"../Tuberculosis/byYear/PMID-{year}.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(sorted(result)))

def main():
    for year in range(2005, 2026):
        save_PMID(year)


if __name__ == "__main__":
    main()
