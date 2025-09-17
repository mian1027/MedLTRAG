import requests
import certifi

# Fetch PubTator annotations for given PMIDs
def fetch_pubtator(pmids: str):
    url = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/pubtator"
    params = {
        "pmids": pmids
    }
    try:
        response = requests.get(url, params=params, verify=certifi.where())
        if response.status_code == 200:
            # Parse plain text response (not JSON)
            text = response.text
            return text
        else:
            print(f"Request failed: HTTP {response.status_code}")
            return None
    except:
        print(f"Request failed: pmids {pmids}")
        return None

# Save PubTator annotations for a given year
def pubTator(year, base_path):
    texts = []
    PMIDs = []
    num = 0

    pmid_file_path = f"{base_path}/PMID-{year}.txt"
    output_pubtator_path = f"{base_path}/PMID_{year}.pubtator"

    with open(pmid_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            print(f"Requesting: {line.strip()}, count: {num}, total PMIDs: {len(PMIDs)}")
            num += 1
            text = fetch_pubtator(pmids=line.strip())
            texts.append(text)
            PMIDs.append(line.strip())

    if len(texts) > 0:
        with open(output_pubtator_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(sorted(str(x) for x in texts if x is not None)))


# Main function
def main():
    for year in range(2005, 2026):
        pubTator(year,"../Tuberculosis/byYear")


if __name__ == "__main__":
    main()