"""Create a focused notebook that repairs product without replacing the old run."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
baseline = json.loads((root / "notebooks/baseline_colab.ipynb").read_text(encoding="utf-8"))
setup = "".join(next(cell["source"] for cell in baseline["cells"] if cell["cell_type"] == "code"))
cells = []


def markdown(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)})


def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "source": source.splitlines(keepends=True),
                  "execution_count": None, "outputs": []})


markdown("""# Oprava produktového BERT modelu — 3 kroky

Tento notebook trénuje **iba `product`**. Ostatné tri modely aj pôvodný chybný
produktový model ponechá na Drive. Nový model bude v `models/product_v2` a nové
predikcie v `predictions/product_v2`. Gemini sa v týchto troch krokoch nevolá.

1. Zastav staré Gemini behy, vyber **SEMEVAL_T4** a reštartuj kernel, aby sa načítal nový kód.
2. Nahraj **nový `baseline_colab.zip`** do `/content/` a potvrď prepísanie starého ZIP-u.
3. Spusti kroky **1 → 2 → 3** nižšie. Pri prerušení zopakuj 1 a 2; tréning sa obnoví.

**Prečo oprava:** produkt má 1 022 výstupov, ale na každej vzorke je len jeden pozitívny.
Pôvodný priemer focal loss prevažne odmeňoval potlačenie negatívnych výstupov.
Pri nulových logits je súhrnná veľkosť negatívnych logitových gradientov 3 063-krát
väčšia než pozitívna. Nová redukcia priemeruje pozitívnu a negatívnu časť osobitne,
stále s binárnou sigmoid focal loss. Produkt sa vyhodnocuje po celej epoche,
checkpoint sa vyberá podľa recall@5 a patience je 3 takéto vyhodnotenia.
""")
markdown("""## 1. Pripojenie a diagnostika
Pripojí existujúci priečinok na Drive a vypíše dostupnú históriu pôvodného modelu.
Použije už pripravené EDA dáta; ostatné modely sa netrénujú.
""")
code(setup + '''
REPAIR_CONFIG = PROJECT / 'configs/product_repair.json'
if not REPAIR_CONFIG.is_file():
    raise RuntimeError('Nahraj novy ZIP obsahujuci configs/product_repair.json.')
required_data = ['training_augmented.jsonl', 'validation.jsonl', 'test.jsonl', 'labels.json']
if any(not (RUN_DIR / 'data' / name).is_file() for name in required_data):
    raise RuntimeError('Na Drive chybaju pripravene data. Over cestu RUN_DIR k povodnemu baseline_v1.')

def run_repair(*args):
    command = [sys.executable, '-u', '-m', 'baseline.cli', '--config', str(REPAIR_CONFIG),
               '--run-dir', str(RUN_DIR), *args]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
        result = process.wait()
    if result:
        raise RuntimeError('Opravny krok zlyhal; dovod je vo vypise vyssie.')

run_repair('diagnose-product')
print('DIAGNOSTIKA HOTOVA. Teraz spusti krok 2.')
''')
markdown("""## 2. Nový tréning iba produktu
Začína od predtrénovaného enkodéra a novej produktovej hlavy. Nepokračuje z chybného
starého modelu. `--resume` obnovuje iba nový beh `product_v2`, ak bol prerušený.
Tréning má najviac 5 epoch, vyváženú binárnu focal loss, alpha 0.5, learning rate 2e-5.
Výber checkpointu sa riadi recall@5, pretože účelom modelu je ponúknuť správneho kandidáta.

Na konci sa automaticky skontroluje, či model nepredikuje len jeden produkt a či má
nenulovú presnosť a recall@5. Ak táto kontrola zlyhá, beh nebude označený ako úspešný.
""")
code('''run_repair('train', '--task', 'product', '--resume')
health = read_json(RUN_DIR / 'models/product_v2/validation_health.json')
print(json.dumps(health, indent=2, ensure_ascii=False))
print('PRODUKT DOTRENOVANY. Spusti krok 3 na kontrolu kandidatov bez Gemini.')
''')
markdown("""## 3. Overenie na validácii — bez Gemini API
Predikcia použije tri pôvodné modely a nový produktový model. Nič neprepisuje
v pôvodnom priečinku `predictions`. Dôležitý je najmä **recall kandidátov produktu**:
musí byť vyšší než pôvodných 0 %. Vyššia hodnota dáva Gemini šancu vybrať správny produkt.
Priemerné skóre nie je zárukou kvality; výsledky si pozri pred ďalšími API volaniami.
""")
code('''run_repair('predict', '--split', 'validation')
REPAIRED_PREDICTIONS = RUN_DIR / 'predictions/product_v2'
run_repair('evaluate', '--split', 'validation', '--predictions',
           str(REPAIRED_PREDICTIONS / 'validation.bert.jsonl'))
candidate_report = read_json(REPAIRED_PREDICTIONS / 'validation.candidate_report.json')
print('NOVY PRODUKTOVY RECALL KANDIDATOV:', candidate_report['product'])
new_metrics = read_json(REPAIRED_PREDICTIONS / 'validation.bert.metrics.json')
print('Nove BERT ST1:', new_metrics['ST1']['score'], 'ST2:', new_metrics['ST2']['score'])
old_path = RUN_DIR / 'predictions/validation.bert.metrics.json'
if old_path.exists():
    old = read_json(old_path)
    print('Povodne BERT ST1:', old['ST1']['score'], 'ST2:', old['ST2']['score'])

# Reuse the original API answers later: unchanged task prompts keep their cache keys.
import shutil
source_audit = RUN_DIR / 'predictions/validation.gemini.jsonl.audit.jsonl'
new_audit = REPAIRED_PREDICTIONS / 'validation.gemini.jsonl.audit.jsonl'
if source_audit.exists() and not new_audit.exists():
    temporary = new_audit.with_suffix(new_audit.suffix + '.tmp')
    shutil.copyfile(source_audit, temporary)
    temporary.replace(new_audit)
    print('Povodna Gemini cache skopirovana pre buduce pouzitie; povodna zostala zachovana.')
print('OVERENIE HOTOVE. Zatial neposielaj nove Gemini poziadavky; najprv skontroluj recall a presnost produktu.')
''')
markdown("""## Čo ďalej

- Nový model: `MyDrive/semeval_task9/baseline_v1/models/product_v2/final/`.
- Diagnostika: `product_diagnostics.json`; metriky a priebeh nového modelu sú v `models/product_v2/`.
- Nové kandidáty a BERT výsledky: `predictions/product_v2/validation.*`.
- Pôvodné tri modely aj pôvodné výsledky zostávajú na svojom mieste.

Pre ďalšie predikcie používaj **`run_repair`**, nie pôvodný `run`: opravná konfigurácia
vyberá nový produktový model. Po overení kvality možno použiť `run_repair('llm', ...)`
nad novými kandidátmi a s výstupom `REPAIRED_PREDICTIONS / 'validation.gemini.jsonl'`.
Pôvodné odpovede ostatných troch cieľov sa môžu znova použiť z cache. Zmenené produktové
prompty potrebujú nové odpovede. Testovacie dáta sa pri tomto rozhodovaní nepoužívajú.
""")
for index, cell in enumerate(cells):
    cell["id"] = f"product-repair-{index}"
notebook = {"cells": cells, "metadata": baseline["metadata"], "nbformat": 4, "nbformat_minor": 5}
path = root / "notebooks/repair_product_colab.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(path)
