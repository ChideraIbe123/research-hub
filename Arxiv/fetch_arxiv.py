import os
import datetime
import requests
import feedparser
import urllib.parse

def fetch_last_weeks_top_papers(folder="arxiv_pdfs"):
    """Fetch the top 10 arXiv papers from the past week (by submittedDate),
       sorted by relevance (descending), and download their PDFs.
    """

    # 1) Create output folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 2) Calculate date range for the last 7 days
    today = datetime.date.today()
    week_ago = today - datetime.timedelta(days=7)

    # 3) Format as YYYYMMDD for the official arXiv API query
    start_str = week_ago.strftime("%Y%m%d")
    end_str   = today.strftime("%Y%m%d")

    # 4) Construct the query parameters.
    params = {
        "search_query": f"submittedDate:[{start_str} TO {end_str}]",
        "sortBy": "relevance",
        "sortOrder": "descending",
        "start": 0,
        "max_results": 10
    }

    # 5) Encode the parameters to ensure the URL is valid
    encoded_params = urllib.parse.urlencode(params)

    # 6) Construct the full URL
    base_url = "http://export.arxiv.org/api/query"
    url = f"{base_url}?{encoded_params}"

    print("Querying URL:", url)

    try:
        # 7) Make the HTTP GET request with a user-agent header
        headers = {'User-Agent': 'arXiv-Paper-Fetcher/1.0'}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed: {e}")
        return

    # 8) Parse the Atom feed using feedparser
    feed = feedparser.parse(response.text)

    # 9) Check total results
    total_found = int(feed.feed.get("opensearch_totalresults", 0))
    print(f"Found {total_found} total results in the last week by submitted date.\n")

    if not feed.entries:
        print("No entries found for that time range. Exiting.")
        return

    # 10) Iterate over each entry (up to 10) and download the PDF
    for idx, entry in enumerate(feed.entries, start=1):
        # Each entry has multiple links; find the one with title="pdf"
        pdf_link = None
        for link in entry.links:
            if link.rel == "related" and link.get("title") == "pdf":
                pdf_link = link.href
                break

        title = entry.title
        # ArXiv ID can be found in entry.id, which might look like "http://arxiv.org/abs/2301.12345v2"
        # Let's just strip out the trailing URL part:
        arxiv_id = entry.id.split("/")[-1]  # e.g. "2301.12345v2"

        print(f"({idx}) Title: {title}")
        if pdf_link:
            print(f"     PDF link: {pdf_link}")
            # Construct a filename from the arXiv ID
            pdf_filename = f"{arxiv_id}.pdf"
            out_path = os.path.join(folder, pdf_filename)

            # Download the PDF
            try:
                with requests.get(pdf_link, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(out_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:  # filter out keep-alive chunks
                                f.write(chunk)
                print(f"     Saved PDF to {out_path}\n")
            except requests.exceptions.RequestException as e:
                print(f"     ERROR downloading PDF: {e}\n")
        else:
            print("     No PDF link found.\n")

def main():
    fetch_last_weeks_top_papers("arxiv_pdfs")

if __name__ == "__main__":
    main()
