"""Create the user's private Colab bundle, including dataset and .env."""
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
target = root / "baseline_colab.zip"
paths = [root / "pyproject.toml", root / "README.md", root / "configs" / "baseline.json",
         root / ".gitignore", root / ".env", root / "dataset" / "food_recall_incidents.csv",
         root / "notebooks" / "baseline_colab.ipynb"]
paths += sorted((root / "baseline").glob("*.py"))
paths += sorted((root / "docs").glob("*.md"))
paths += sorted((root / "scripts").glob("*.py"))
paths += sorted((root / "tests").glob("*.py"))
missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
if missing:
    raise FileNotFoundError(f"Missing required bundle files: {', '.join(missing)}")
with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in paths:
        archive.write(path, path.relative_to(root).as_posix())
print(f"Created {target} ({target.stat().st_size / 1024**2:.1f} MiB); includes .env, dataset and notebook. Keep this bundle private.")
