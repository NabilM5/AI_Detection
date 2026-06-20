import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from urllib.parse import urljoin


INDEX_URL = "https://medlineplus.gov/xml.html"
RAW_DIR = "data/external/medlineplus"
RAW_XML_PATH = os.path.join(RAW_DIR, "mplus_topics.xml")
OUTPUT_JSONL_PATH = "data/processed/medlineplus_human.jsonl"


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def clean_html(text):
    if text is None:
        return ""

    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())
    return text.strip()


def find_latest_xml_url(index_url):
    print(f"Finding latest XML file from: {index_url}")

    with urllib.request.urlopen(index_url) as response:
        html = response.read().decode("utf-8")

    parser = LinkParser()
    parser.feed(html)

    xml_links = [
        urljoin(index_url, href)
        for href in parser.links
        if re.search(r"/xml/mplus_topics_\d{4}-\d{2}-\d{2}\.xml$", href)
    ]

    if not xml_links:
        raise RuntimeError(
            f"No MedlinePlus health topic XML links found at {index_url}"
        )

    return max(xml_links)


def download_file(url, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Downloading from: {url}")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved raw XML to: {output_path}")


def parse_medlineplus_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    records = []

    for topic in root.findall(".//health-topic"):
        language = topic.attrib.get("language")
        topic_id = topic.attrib.get("id")
        title = topic.attrib.get("title")
        url = topic.attrib.get("url")

        if language != "English":
            continue

        full_summary_el = topic.find("full-summary")
        summary = clean_html(
            "".join(full_summary_el.itertext()) if full_summary_el is not None else ""
        )

        if not summary:
            continue

        record = {
            "id": f"medlineplus_{topic_id}_human",
            "text": summary,
            "label": "human",
            "source_dataset": "medlineplus",
            "source_document_id": f"medlineplus_{topic_id}",
            "genre": "patient_education",
            "medical_topic": title,
            "url": url,
            "license": "medlineplus_attribution_required",
            "generator": None,
            "prompt_template": None,
            "split_group": f"medlineplus_{topic_id}",
        }

        records.append(record)

    return records


def write_jsonl(records, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} records to: {output_path}")


def main():
    latest_xml_url = find_latest_xml_url(INDEX_URL)
    download_file(latest_xml_url, RAW_XML_PATH)

    records = parse_medlineplus_xml(RAW_XML_PATH)
    write_jsonl(records, OUTPUT_JSONL_PATH)

    print("\nPreview:")
    for record in records[:3]:
        print("-" * 40)
        print(record["medical_topic"])
        print(record["text"][:300], "...")


if __name__ == "__main__":
    main()
