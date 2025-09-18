import tomllib
from pathlib import Path

# الحزم المطلوبة
required = [
    "transformers",
    "datasets",
    "tokenizers",
    "huggingface-hub",
    "torch",
    "scikit-learn",
    "pandas",
    "numpy",
    "evaluate",
]

pyproject = Path("pyproject.toml")
if not pyproject.exists():
    raise FileNotFoundError("pyproject.toml مش موجود في هذا المجلد.")

# اقرأ التومال
data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

deps = data.get("project", {}).get("dependencies", [])

print("== فحص dependencies في pyproject.toml ==\n")
for pkg in required:
    found = any(dep.lower().startswith(pkg.lower()+"==") or dep.lower()==pkg.lower() for dep in deps)
    mark = "✅" if found else "❌"
    print(f"{mark} {pkg}")
