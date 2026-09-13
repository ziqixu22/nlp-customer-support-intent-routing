from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"

def main():
    out = Path("data")
    out.mkdir(exist_ok=True)
    urlretrieve(f"{BASE}/train.csv", out / "train.csv")
    urlretrieve(f"{BASE}/test.csv", out / "test.csv")
    print("Downloaded BANKING77 train/test CSV files")

if __name__ == "__main__":
    main()
