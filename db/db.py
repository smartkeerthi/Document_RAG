import json
from pathlib import Path

DB_DIR = Path("db")
DB_DIR.mkdir(exist_ok=True)
DB_NAME = "db.json"
dbPath = DB_DIR/DB_NAME


def readDb() -> dict[str, str]:
    if not dbPath.exists():
        writeDb({})
    
    with dbPath.open("r", encoding="utf-8") as f:
        return json.load(f)
    

def writeDb(data: dict[str, str]):
    with dbPath.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)