import json
import os
import sys
import time
from datetime import date
from pathlib import Path

from openai import AuthenticationError
from openai import RateLimitError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env import load_dotenv
from src.utils.llm_client import LLMClient
from src.utils.llm_client import LLMAPIError


load_dotenv(PROJECT_ROOT / ".env")

INPUT_PATH = "data/processed/medlineplus_human.jsonl"
OUTPUT_PATH = "data/processed/medlineplus_ai.jsonl"

MODEL_NAME = os.getenv("LLM_MODEL", "unknown_model")
MAX_TOPICS = int(os.getenv("MAX_TOPICS", "100"))
TARGET_TOTAL_AI_RECORDS = int(os.getenv("TARGET_TOTAL_AI_RECORDS", "100"))
REQUEST_DELAY_SECONDS = float(os.getenv("LLM_REQUEST_DELAY_SECONDS", "15"))
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
MIN_OUTPUT_WORDS = int(os.getenv("MIN_OUTPUT_WORDS", "80"))


def log(message=""):
    print(message, flush=True)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_existing_ids(path):
    if not os.path.exists(path):
        return set()

    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ids.add(json.loads(line)["source_document_id"])

    return ids


def count_jsonl_records(path):
    if not os.path.exists(path):
        return 0

    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def build_prompt(topic, human_text):
    return f"""
You are writing a patient-facing medical education article.

Topic: {topic}

Write a clear, factual, patient-friendly explanation about this topic.
Do not mention that you are an AI.
Do not copy the source text.
Use neutral medical language.
Keep the answer between 150 and 300 words.
"""


def generate_ai_text(prompt):
    return client.generate_text(prompt, temperature=0.7)


def is_too_short(text):
    return len(text.split()) < MIN_OUTPUT_WORDS


def generate_with_retries(topic, prompt):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return generate_ai_text(prompt)
        except LLMAPIError as e:
            if e.status_code in {429, 503}:
                wait_seconds = e.retry_after or REQUEST_DELAY_SECONDS * attempt
                log(
                    f"Temporary {e.provider} API error {e.status_code} on {topic}; "
                    f"waiting {wait_seconds:.0f}s before retry {attempt}/{MAX_RETRIES}."
                )
                time.sleep(wait_seconds)
                continue

            raise

    log(f"Skipped {topic}: failed after {MAX_RETRIES} retries.")
    return None


def main():
    global client

    try:
        client = LLMClient()
    except RuntimeError as e:
        log(e)
        return

    human_records = load_jsonl(INPUT_PATH)
    existing_ids = load_existing_ids(OUTPUT_PATH)
    existing_count = count_jsonl_records(OUTPUT_PATH)

    if existing_count >= TARGET_TOTAL_AI_RECORDS:
        log(
            f"{OUTPUT_PATH} already contains {existing_count} AI records; "
            f"target is {TARGET_TOTAL_AI_RECORDS}."
        )
        return

    generated_count = 0
    total_count = existing_count
    target_new_records = min(MAX_TOPICS, TARGET_TOTAL_AI_RECORDS - existing_count)

    log(
        f"Starting with {existing_count} existing AI records. "
        f"Generating up to {target_new_records} more to reach "
        f"{TARGET_TOTAL_AI_RECORDS} total."
    )

    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        for record in human_records:
            if total_count >= TARGET_TOTAL_AI_RECORDS:
                break

            source_document_id = record["source_document_id"]

            if source_document_id in existing_ids:
                continue

            topic = record["medical_topic"]
            prompt = build_prompt(topic, record["text"])

            log(f"Generating: {topic}")

            try:
                ai_text = generate_with_retries(topic, prompt)
            except AuthenticationError as e:
                log("\nStopped: API authentication failed.")
                log("Check that LLM_API_KEY in .env is set to a real key.")
                log(f"API error: {e}")
                break
            except RateLimitError as e:
                if getattr(e, "code", None) == "insufficient_quota":
                    log(
                        "\nStopped: API quota is exhausted or billing is not active "
                        "for this API key."
                    )
                    log("Check usage and billing, then rerun this script.")
                    break

                log(f"Rate limit error on {topic}: {e}")
                time.sleep(10)
                continue
            except Exception as e:
                log(f"Error on {topic}: {e}")
                continue

            if ai_text is None:
                continue

            if is_too_short(ai_text):
                log(f"Skipped {topic}: output has fewer than {MIN_OUTPUT_WORDS} words.")
                continue

            ai_record = {
                "id": record["id"].replace("_human", "_ai_generated"),
                "text": ai_text,
                "label": "ai",
                "source_dataset": "medlineplus_llm_generated",
                "source_document_id": source_document_id,
                "genre": "patient_education",
                "medical_topic": topic,
                "source_human_url": record["url"],
                "license": "generated_from_medlineplus_topic_prompt",
                "generator": MODEL_NAME,
                "generation_date": str(date.today()),
                "temperature": 0.7,
                "prompt_template": "patient_facing_medical_education_150_300_words",
                "split_group": source_document_id,
            }

            f.write(json.dumps(ai_record, ensure_ascii=False) + "\n")
            f.flush()

            generated_count += 1
            total_count += 1
            existing_ids.add(source_document_id)

            if total_count % 10 == 0 or total_count == TARGET_TOTAL_AI_RECORDS:
                log(
                    f"Progress: {total_count}/{TARGET_TOTAL_AI_RECORDS} total AI records "
                    f"({generated_count} generated this run)."
                )

            time.sleep(REQUEST_DELAY_SECONDS)

    log(f"\nGenerated {generated_count} AI records.")
    log(f"Total AI records now: {total_count}")
    log(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
