"""Run in a Colab cell with exec(Path(...).read_text()); works with older baseline clients too."""
import sys
import time
from pathlib import Path

project = Path("/content/semeval_baseline")
sys.path.insert(0, str(project))
from baseline import llm
from baseline.io import config_at
from baseline.metrics import evaluate

run_dir = Path("/content/drive/MyDrive/semeval_task9/baseline_v1")
predictions = run_dir / "predictions"
config = config_at(project / "configs/baseline.json", run_dir)

# Patch the actual HTTP function in this process, not in an unrelated subprocess.
# Keep generation settings unchanged so previously cached answers remain valid.
original_urlopen = llm.urllib.request.urlopen
last_request = [None]


def limited_urlopen(*args, **kwargs):
    if last_request[0] is not None:
        delay = 13 - (time.monotonic() - last_request[0])
        if delay > 0:
            print(f"Cakam {delay:.1f} s pred dalsim HTTP pokusom...", flush=True)
            time.sleep(delay)
    last_request[0] = time.monotonic()
    print("Gemini HTTP pokus:", time.strftime("%H:%M:%S"), flush=True)
    return original_urlopen(*args, **kwargs)


print("Zastav ostatne Gemini behy. Cakam 60 s na uvolnenie predchadzajuceho okna...", flush=True)
time.sleep(60)
llm.urllib.request.urlopen = limited_urlopen
try:
    llm.llm_predict(config, predictions / "validation.candidates.jsonl",
                    predictions / "validation.gemini.jsonl")
finally:
    llm.urllib.request.urlopen = original_urlopen

scores = evaluate(config, "validation", predictions / "validation.gemini.jsonl")
print(f"KROK 3 HOTOVY. ST1 = {scores['ST1']['score']:.4f}; ST2 = {scores['ST2']['score']:.4f}")
