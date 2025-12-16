import os
import json

INPUT_FILE="rag/TimetableData/data.json"
OUTPUT_FOLDER="rag/TimetableData/processed"

def load_json():
    with open(INPUT_FILE, "r") as f:
        return json.load(f)
    
def ensure_output_dir():
    os.makedirs("rag/TimetableData/processed", exist_ok=True)

def normalize_text(text: str):
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    return " ".join(text.split(" "))

def convert_record_to_text(record):
    parts = [
        f"The subject '{record.get('subjectFullName')}' (subject code: {record.get('subjectCode')}) ",
        f"is offered by the {record.get('offeringDept')} department for {record.get('year')} year ",
        f"{record.get('degree')} students. It carries {record.get('subjectCredit')} credits and ",
        f"is categorized as a '{record.get('subjectType')}' course. ",
        f"The class is conducted by faculty member {record.get('faculty')} on {record.get('day')} ",
        f"in room {record.get('room')}, during the {record.get('slot')} slot. ",
        f"This course is scheduled for the {record.get('sem')} semester of the {record.get('session')} session. ",
    ]

    return normalize_text(" ".join(parts))

def preprocess():
    ensure_output_dir()

    data = load_json()
    chunks = []

    for record in data:
        text = convert_record_to_text(record)
        if text:
            chunks.append({"text":text})
        
    OUTPUT_FILE=f"{OUTPUT_FOLDER}/clean_chunks.json"

    with open(OUTPUT_FILE, "w") as f:
        json.dump(chunks, f, indent=4)

    print(f"Preprocessing complete. {len(chunks)} chunks written to {OUTPUT_FILE}")