from pathlib import Path

# backend root
ROOT = Path(__file__).parent.parent

DATA_PATH = ROOT.parent.parent / "data"


def load_filenames(filepath: Path) -> list[str]:
    with open(filepath) as f:
        filenames = f.readlines()
        return [filename.strip() for filename in filenames]
