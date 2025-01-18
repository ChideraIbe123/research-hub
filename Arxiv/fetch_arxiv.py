import os
import datetime
import requests
import feedparser
import urllib.parse
from pdf2image import convert_from_path
from PIL import Image


def fetch_last_weeks_top_papers(pdf_folder="arxiv_pdfs", info_folder="arxiv_info", images_folder="./website/images"):
    """
    Fetch the top 3 arXiv papers from the past week (by submittedDate),
    sorted by relevance (descending), download their PDFs, and save
    metadata + a first-page preview image. Metadata includes: title,
    abstract, authors, primary category, PDF link, the arXiv abstract link, and publication date.
    """

    # 1. Prepare folders for PDFs, info files, and images
    # ------------------------------------------
    # Remove old PDFs (if the folder exists), then re-create it empty
    if os.path.exists(pdf_folder):
        for filename in os.listdir(pdf_folder):
            file_path = os.path.join(pdf_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(pdf_folder)

    # Create or empty the info folder
    if os.path.exists(info_folder):
        for filename in os.listdir(info_folder):
            file_path = os.path.join(info_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(info_folder)

    # Create images folder if it doesn't exist
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    # 2. Build arXiv API query for last week
    # --------------------------------------
    today = datetime.date.today()
    week_ago = today - datetime.timedelta(days=1)

    start_str = week_ago.strftime("%Y%m%d")
    end_str   = today.strftime("%Y%m%d")

    params = {
        "search_query": f"submittedDate:[{start_str} TO {end_str}]",
        "sortBy": "relevance",
        "sortOrder": "descending",
        "start": 0,
        "max_results": 1
    }

    encoded_params = urllib.parse.urlencode(params)
    base_url = "http://export.arxiv.org/api/query"
    url = f"{base_url}?{encoded_params}"

    print("Querying URL:", url)

    # 3. Perform request to arXiv
    # ---------------------------
    try:
        headers = {'User-Agent': 'arXiv-Paper-Fetcher/1.0'}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed: {e}")
        return

    # 4. Parse feed and retrieve entries
    # ----------------------------------
    feed = feedparser.parse(response.text)
    total_found = int(feed.feed.get("opensearch_totalresults", 0))
    print(f"Found {total_found} total results in the last week by submitted date.\n")

    if not feed.entries:
        print("No entries found for that time range. Exiting.")
        return

    # 5. Process each entry: download PDF, save metadata, create overlay image
    # ------------------------------------------------------------------------
    category_mapping = {
        "physics": "Physics",
        "math": "Mathematics",
        "cs": "Computer Science",
        "q-bio": "Quantitative Biology",
        "stat": "Statistics",
        "eess": "Electrical Engineering and Systems Science",
        "econ": "Economics"
    }

    for idx, entry in enumerate(feed.entries, start=1):
        # Extract PDF link
        pdf_link = None
        for link in entry.links:
            if link.rel == "related" and link.get("title") == "pdf":
                pdf_link = link.href
                break

        # Extract metadata
        title      = entry.title.strip()
        abstract   = entry.summary.strip()
        authors    = [author.name for author in entry.authors] if hasattr(entry, 'authors') else []
        categories = [t['term'] for t in entry.tags] if hasattr(entry, 'tags') else []
        primary_category = categories[0] if categories else "N/A"
        published_date = entry.published if hasattr(entry, 'published') else "N/A"
        
        # Map primary category to one of the specified categories
        primary_category_mapped = "N/A"
        for key in category_mapping:
            if primary_category.startswith(key):
                primary_category_mapped = category_mapping[key]
                break
        
        # This link is typically the arXiv abstract page (e.g., https://arxiv.org/abs/...)
        abstract_link = entry.link if hasattr(entry, 'link') else None

        # arXiv ID used to build filenames
        arxiv_id = entry.id.split("/")[-1]

        print(f"({idx}) Title: {title}")

        if pdf_link:
            print(f"     PDF link: {pdf_link}")
            pdf_filename = f"{arxiv_id}.pdf"
            out_path = os.path.join(pdf_folder, pdf_filename)

            # Download the PDF
            try:
                with requests.get(pdf_link, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(out_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                print(f"     Saved PDF to {out_path}")

            except requests.exceptions.RequestException as e:
                print(f"     ERROR downloading PDF: {e}")
                continue  # Skip to the next paper

            # Save metadata (including the abstract link) to a text file
            metadata_filename = f"{arxiv_id}.txt"
            metadata_path = os.path.join(info_folder, metadata_filename)
            with open(metadata_path, "w", encoding="utf-8") as meta_file:
                meta_file.write(f"Title: {title}\n")
                meta_file.write("Authors:\n")
                for author in authors:
                    meta_file.write(f"  - {author}\n")
                meta_file.write(f"Primary Category: {primary_category_mapped}\n")
                meta_file.write(f"PDF Link: {pdf_link}\n")
                if abstract_link:
                    meta_file.write(f"Abstract Link: {abstract_link}\n")
                meta_file.write(f"Published Date: {published_date}\n")
                meta_file.write("\nAbstract:\n")
                meta_file.write(abstract + "\n")

            print(f"     Saved metadata to {metadata_path}")

            # Convert first page of PDF to image
            try:
                pages = convert_from_path(out_path, first_page=1, last_page=1)
                if pages:
                    first_page_image = pages[0]

                    # Save this image in the images folder
                    image_filename = f"{arxiv_id}.png"
                    image_path_website = os.path.join(images_folder, image_filename)
                    image_path_info = os.path.join(info_folder, image_filename)

                    first_page_image.save(image_path_website)
                    first_page_image.save(image_path_info)

                    print(f"     Saved first-page image to {image_path_website}\n")
                    print(f"     Also saved first-page image to {image_path_info}\n")

            except Exception as e:
                print(f"     ERROR converting PDF to image: {e}\n")

        else:
            print("     No PDF link found.\n")


def main():
    fetch_last_weeks_top_papers("arxiv_pdfs", "arxiv_info")


if __name__ == "__main__":
    main()
