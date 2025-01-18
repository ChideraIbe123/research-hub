import os
import re
from datetime import datetime
from pathlib import Path

# Directories
ARXIV_INFO_DIR = Path('./arxiv_info')
SUMMARIES_DIR = Path('./fixed_summaries')  # Updated to point to fixed_summaries
IMPACT_DIR = Path('./impact')  # Directory for impact summaries
PAPERS_TS_PATH = Path('./website/src/data/papers.ts')  # Update this path accordingly

# Regular expressions to parse the fields
FIELD_REGEX = {
    'Title': re.compile(r'^Title:\s*(.+)$', re.MULTILINE),
    'Authors': re.compile(r'^Authors:\s*((?:\s*-\s*.+\n?)+)', re.MULTILINE),
    'Primary Category': re.compile(r'^Primary Category:\s*(.+)$', re.MULTILINE),
    'PDF Link': re.compile(r'^PDF Link:\s*(.+)$', re.MULTILINE),
    'Published Date': re.compile(r'^Published Date:\s*(.+)$', re.MULTILINE),
    'Abstract': re.compile(r'^Abstract:\s*([\s\S]+)$', re.MULTILINE),
}

def parse_authors(authors_block):
    """Parse the authors from the authors block."""
    authors = re.findall(r'-\s*(.+)', authors_block)
    return authors

def parse_published_date(date_str):
    """
    Parse the published date from ISO 8601 format to datetime object.
    Example: '2024-12-29T00:11:22Z' -> datetime(2024, 12, 29, 0, 11, 22)
    """
    try:
        # Replace 'Z' with '+00:00' to make it ISO 8601 compliant for fromisoformat
        if date_str.endswith('Z'):
            date_str = date_str.replace('Z', '+00:00')
        date = datetime.fromisoformat(date_str)
        return date
    except ValueError:
        print(f"Warning: Invalid date format '{date_str}'. Using current date.")
        return datetime.now()

def parse_summary(file_path):
    """Parse the summary from the fixed_summaries folder."""
    summary_file = SUMMARIES_DIR / f"fixed_{file_path.stem}_summary.txt"  # Updated to look in fixed_summaries
    if not summary_file.exists():
        print(f"Warning: Summary file {summary_file} does not exist. Using abstract as summary.")
        return None
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary = f.read().strip().replace('\n', ' ')
    return summary

def parse_impact(file_path):
    """Parse the impact summary from the impact folder."""
    impact_file = IMPACT_DIR / f"impact_fixed_{file_path.stem}_summary.txt"
    if not impact_file.exists():
        print(f"Warning: Impact file {impact_file} does not exist. Skipping impact for this paper.")
        return None
    with open(impact_file, 'r', encoding='utf-8') as f:
        impact = f.read().strip()  # Remove .replace('\n', ' ')
    return impact


def parse_image(file_path):
    """Get the image path from the arxiv_info folder."""
    image_file = ARXIV_INFO_DIR / f"{file_path.stem}.png"
    if not image_file.exists():
        print(f"Warning: Image file {image_file} does not exist. Skipping image for this paper.")
        return None
    # Assuming images are served from '/images/' directory in your website
    # Adjust the path as necessary based on your project's structure
    return f"../images/{image_file.name}"

def parse_arxiv_file(file_path):
    """Parse a single arXiv info text file and extract paper details."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Extract fields using regex
    title_match = FIELD_REGEX['Title'].search(content)
    authors_match = FIELD_REGEX['Authors'].search(content)
    category_match = FIELD_REGEX['Primary Category'].search(content)
    pdf_link_match = FIELD_REGEX['PDF Link'].search(content)
    published_date_match = FIELD_REGEX['Published Date'].search(content)
    abstract_match = FIELD_REGEX['Abstract'].search(content)

    if not all([title_match, authors_match, category_match, pdf_link_match, published_date_match, abstract_match]):
        print(f"Warning: Missing fields in {file_path.name}. Skipping.")
        return None

    title = title_match.group(1).strip()
    authors = parse_authors(authors_match.group(1))
    primary_category = category_match.group(1).strip()
    pdf_link = pdf_link_match.group(1).strip()
    abstract = abstract_match.group(1).strip().replace('\n', ' ')

    published_date_str = published_date_match.group(1).strip()
    date = parse_published_date(published_date_str)

    # Get summary
    summary = parse_summary(file_path)
    if not summary:
        summary = abstract  # Fallback to abstract if summary is missing

    # Get impact
    impact = parse_impact(file_path)

    # Get image path
    image = parse_image(file_path)

    # Construct the journal field (e.g., "Mathematics")
    journal = f"{primary_category}"

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "summary": summary,
        "impact": impact,
        "date": date.strftime('"%Y-%m-%d"'),  # Correct format for TypeScript
        "journal": journal,
        "link": pdf_link,
        "image": image,
    }

def generate_papers_ts(papers):
    """Generate the content for papers.ts based on the list of papers."""
    ts_lines = [
        "export const papers = [\n"
    ]

    for paper in papers:
        # Handle authors array
        authors_formatted = ', '.join([f'"{author}"' for author in paper['authors']])
        authors_ts = f'[{authors_formatted}]'

        # Handle image field
        image_ts = f'`{paper["image"]}`' if paper["image"] else 'null'

        # Handle impact field
        impact_ts = f'`{paper["impact"]}`' if paper["impact"] else 'null'

        paper_entry = f"""  {{
    title: `{paper['title']}`,
    authors: {authors_ts},
    abstract: `{paper['abstract']}`,
    summary: `{paper['summary']}`,
    impact: {impact_ts},
    date: new Date({paper['date']}),
    journal: `{paper['journal']}`,
    link: `{paper['link']}`,
    image: {image_ts},
  }},\n"""
        ts_lines.append(paper_entry)

    ts_lines.append("];\n")

    return ''.join(ts_lines)

def main():
    """Main function to parse all text files and update papers.ts."""
    papers = []
    for txt_file in ARXIV_INFO_DIR.glob('*.txt'):
        paper = parse_arxiv_file(txt_file)
        if paper:
            papers.append(paper)

    if not papers:
        print("No valid papers found. Exiting.")
        return

    # Sort papers by date descending
    # Convert date string back to datetime for sorting
    for paper in papers:
        paper['parsed_date'] = datetime.strptime(paper['date'].strip('"'), "%Y-%m-%d")
    papers.sort(key=lambda x: x['parsed_date'], reverse=True)
    for paper in papers:
        del paper['parsed_date']

    # Generate the TypeScript content
    papers_ts_content = generate_papers_ts(papers)

    # Write to papers.ts
    with open(PAPERS_TS_PATH, 'w', encoding='utf-8') as ts_file:
        ts_file.write(papers_ts_content)

    print(f"Successfully updated {PAPERS_TS_PATH}")

if __name__ == "__main__":
    main()
