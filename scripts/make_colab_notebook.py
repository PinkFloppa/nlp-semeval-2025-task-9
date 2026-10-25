"""Generate the four-step Colab notebook."""
import json
from pathlib import Path

cells = []


def markdown(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.splitlines(keepends=True)})


markdown("""# Baseline — 4 kroky

Vyber kernel **SEMEVAL_T4** a nahraj aktuálny **baseline_colab.zip** do `/content/`
(pravý klik na ZIP → Upload to Colab). ZIP obsahuje dataset aj `.env` s tvojím API
kľúčom, preto ho uchovaj súkromne.

**Teraz spusti krok 1 a potom krok 2. Počas tréningu počkaj.**
Keď krok 2 vypíše **TRÉNING DOKONČENÝ — 4/4 MODELY**, spusti krok 3.
Krok 4 je finálne vyhodnotenie na teste a spúšťa sa až po kontrole validácie.
**Nepoužívaj Run All**, ak chceš najprv vidieť validačné výsledky.

| Krok | Čo urobí | Kedy ho spustiť |
|---|---|---|
| 1 | Pripojí GPU, rozbalí ZIP a pripojí Drive | Na začiatku každej relácie |
| 2 | Pripraví dáta, EDA a natrénuje 4 modely | Po kroku 1; toto trvá najdlhšie |
| 3 | BERT vyberie kandidátov, Gemini odpovede; zobrazí validačné skóre | Po dokončení tréningu |
| 4 | Získa finálne testovacie skóre a grafy | Keď sú nastavenia uzavreté |

Pri prerušení Colabu zopakuj **1 → 2**. Hotové modely sa preskočia a rozpracovaný
tréning pokračuje z posledného checkpointu. Gemini odpovede sa tiež priebežne ukladajú.
""")
markdown("""## 1. Pripojiť a pripraviť prostredie
Spusti túto bunku a povoľ pripojenie Google Drive. API kľúč sa načíta z pribaleného
`.env` bez vypísania hodnoty. Výsledky zostávajú na Drive v
`MyDrive/semeval_task9/baseline_v1`, aj keď Colab relácia skončí.
""")
code('''import os, sys, json, subprocess, zipfile
from pathlib import Path

if not Path('/content').is_dir():
    raise RuntimeError('Vyber kernel SEMEVAL_T4, nie lokalny Python.')
os.chdir('/content')
archive = Path('/content/baseline_colab.zip')
if not archive.exists():
    from google.colab import files
    files.upload()  # Vyber aktualny baseline_colab.zip.
if not archive.exists():
    raise FileNotFoundError('Nahraj baseline_colab.zip do /content/.')
PROJECT = Path('/content/semeval_baseline')
PROJECT.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as package:
    required = {'.env', 'dataset/food_recall_incidents.csv', 'configs/baseline.json'}
    if not required.issubset(package.namelist()):
        raise ValueError('Toto je starsi alebo neuplny ZIP. Nahraj aktualny baseline_colab.zip.')
    for name in package.namelist():
        if not (PROJECT / name).resolve().is_relative_to(PROJECT.resolve()):
            raise ValueError('Neplatna cesta v archive.')
    package.extractall(PROJECT)
os.chdir(PROJECT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.'], check=True)
subprocess.run([sys.executable, '-m', 'nltk.downloader', '-q', 'wordnet', 'omw-1.4'], check=True)

from dotenv import dotenv_values
key = dotenv_values(PROJECT / '.env').get('GEMINI_API_KEY')
if not key:
    raise RuntimeError('V pribalenom .env chyba GEMINI_API_KEY. Dopln ho lokalne a vytvor novy ZIP.')
os.environ['GEMINI_API_KEY'] = key
del key
import torch
if not torch.cuda.is_available():
    raise RuntimeError('GPU nie je dostupna. Vyber T4 runtime.')
print('GPU:', torch.cuda.get_device_name(0))

from google.colab import drive
drive.mount('/content/drive')
RUN_DIR = Path('/content/drive/MyDrive/semeval_task9/baseline_v1')
RUN_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS = RUN_DIR / 'predictions'
from baseline.io import config_at, read_json, write_json
from baseline.training import environment
CONFIG = config_at(PROJECT / 'configs/baseline.json', RUN_DIR)
write_json(RUN_DIR / 'environment.json', environment())
with (RUN_DIR / 'pip-freeze.txt').open('w') as stream:
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
def run(*args):
    command = [sys.executable, '-u', '-m', 'baseline.cli', '--run-dir', str(RUN_DIR), *args]
    # Forward child output explicitly: inherited file descriptors may point to
    # the remote kernel log instead of the VS Code notebook output.
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
def show_scores(split):
    report = read_json(PREDICTIONS / f'{split}.gemini.metrics.json')
    print(f"\\n{split.upper()}: ST1 = {report['ST1']['score']:.4f}; ST2 = {report['ST2']['score']:.4f}")
    print('Pocet vyhodnotenych vzoriek:', report['samples'])
print('KROK 1 HOTOVY. Spusti krok 2. API kluc je nacitany; jeho hodnota sa nevypisuje.')
''')
markdown("""## 2. Pripraviť dáta a natrénovať všetky 4 modely
**Spusti raz a počkaj, kým bunka skončí.** Najprv vzniknú anglické splity a EDA dáta,
potom sa postupne trénujú kategória hazardu, kategória produktu, hazard a produkt.
Logy počas tréningu sú priebežné údaje; finálne ST1/ST2 skóre uvidíš až v krokoch 3 a 4.
Ak bola relácia prerušená, tá istá bunka obnoví rozpracovaný tréning.
""")
code('''from baseline import TASKS
from baseline.data import prepare
from baseline.eda import augment

report = prepare(CONFIG)
write_json(RUN_DIR / 'config.snapshot.json', CONFIG)
print('Rozdelenie dat:', report['split_counts'], flush=True)
eda = augment(CONFIG)
print('EDA:', eda['synthetic'], 'novych vzoriek; spolu', eda['total'], flush=True)
for index, task in enumerate(TASKS, 1):
    print(f'\\n===== MODEL {index}/4: {task} =====', flush=True)
    run('train', '--task', task, '--resume')
print('\\nTRENING DOKONCENY — 4/4 MODELY. Teraz spusti krok 3.', flush=True)
''')
markdown("""## 3. Vyhodnotiť validáciu pomocou BERT + Gemini
Spusti až po dokončení kroku 2. BERT vytvorí kandidátov a **Gemini API z nich vyberie
finálne označenia**. Toto už netrénuje modely. Pri prvom plnom behu vznikne najviac
2 260 API požiadaviek; pri opakovaní sa použijú uložené odpovede. Rýchlosť určuje
`requests_per_minute` v konfigurácii, takže aj tento krok môže trvať niekoľko hodín.
Na konci bunka vypíše validačné **ST1 a ST2 skóre**.

Ak chceš pred plnou validáciou skontrolovať iba 25 vzoriek, použi pilotné príkazy
z README. Pri tomto jednoduchom postupe sa validácia vykoná celá.
""")
code('''run('predict', '--split', 'validation')
run('evaluate', '--split', 'validation', '--predictions', str(PREDICTIONS / 'validation.bert.jsonl'))
run('llm', '--candidates', str(PREDICTIONS / 'validation.candidates.jsonl'),
    '--output', str(PREDICTIONS / 'validation.gemini.jsonl'))
run('evaluate', '--split', 'validation', '--predictions', str(PREDICTIONS / 'validation.gemini.jsonl'))
show_scores('validation')
print('KROK 3 HOTOVY. Po kontrole validacie a uzamknuti nastaveni mozes spustit krok 4.')
''')
markdown("""## 4. Finálny test a výsledky do správy
Spusti až po kontrole validácie, keď už nebudeš meniť nastavenia baseline.
Modely sa znovu netrénujú. Rovnaká pipeline spracuje 997 testovacích vzoriek,
čo znamená najviac 3 988 nových Gemini požiadaviek. Test sa nepoužíva na ladenie.
Na konci uvidíš finálne skóre a dve konfúzne matice ST1.
""")
code('''run('predict', '--split', 'test')
run('evaluate', '--split', 'test', '--predictions', str(PREDICTIONS / 'test.bert.jsonl'))
run('llm', '--candidates', str(PREDICTIONS / 'test.candidates.jsonl'),
    '--output', str(PREDICTIONS / 'test.gemini.jsonl'))
run('evaluate', '--split', 'test', '--predictions', str(PREDICTIONS / 'test.gemini.jsonl'))
show_scores('test')
from IPython.display import Image, display
for task in ('hazard-category', 'product-category'):
    display(Image(filename=str(PREDICTIONS / f'test.gemini.{task}.png')))
print('HOTOVO. Vysledky a modely su ulozene v:', RUN_DIR)
''')
markdown("""## Čo zostane po dokončení

V Google Drive otvor **MyDrive → semeval_task9 → baseline_v1**:

| Súbor alebo priečinok | Na čo slúži |
|---|---|
| `models/<cieľ>/final/` | Štyri natrénované modely; uchovaj ich, netreba trénovať znova |
| `predictions/test.gemini.metrics.json` | Finálne ST1/ST2 skóre a podrobné metriky do správy |
| `predictions/test.bert.metrics.json` | Výsledok samotného BERT-u na porovnanie s celou pipeline |
| `predictions/test.gemini.*.png` | Konfúzne matice ST1 do PDF |
| `predictions/test.candidate_report.json` | Recall a počet kandidátov pre každý cieľ |
| `predictions/test.gemini.jsonl` | Finálne predikcie každej testovacej vzorky |
| `predictions/test.gemini.jsonl.audit.jsonl` | Prompty, odpovede Gemini a spotreba tokenov |
| `data/augmentation.json` a `models/<cieľ>/run.json` | Skutočné počty a parametre pre doplnenie metodiky |

**Do PDF doplň namerané skóre, parametre, candidate recall a konfúzne matice.**
Výsledky porovnaj s referenčnými systémami a opíš obmedzenie nevidených produktových
tried. Záznam použitia AI je v `docs/ai_usage.md`; prompty riešenia sú v auditných logoch.
Po uložení výsledkov môžeš odpojiť Colab. Modely a výsledky na Drive zostanú zachované.
""")

for index, cell in enumerate(cells):
    cell['id'] = f'baseline-{index:02d}'
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
root = Path(__file__).resolve().parent.parent
path = root / 'notebooks/baseline_colab.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(path)
