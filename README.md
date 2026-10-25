# SemEval-2025 Task 9 — baseline

Implementácia podľa kapitoly 4 dokumentu `SemEval_2025_Task_9_1_main_no_parameters.pdf`:
**EDA → štyri samostatné enkodéry s binárnou focal loss → kandidáti → Gemini bez dolaďovania**.

Predvolený enkodér je `FacebookAI/roberta-base` (rodina BERT, podľa návrhu SRCB),
finálny LLM je `gemini-3.5-flash-lite`. Všetky parametre sú v
[`configs/baseline.json`](configs/baseline.json). Modely sa trénujú postupne.

## Spustenie v Colabe / VS Code

1. Lokálne vytvorte prenosný balík: `python scripts/package_colab.py`.
2. Otvorte [`notebooks/baseline_colab.ipynb`](notebooks/baseline_colab.ipynb) a vyberte
   **SEMEVAL_T4**. Pripojenie a spúšťanie buniek prebieha vo VS Code; miestny terminál
   automaticky nepoužíva vzdialenú GPU.
3. Nahrajte `baseline_colab.zip` cez pravý klik **Upload to Colab** do `/content/`,
   prípadne použite upload v prvej bunke notebooku. Balík obsahuje zdrojové súbory,
   konfiguráciu, notebook, **`.env` aj `dataset/food_recall_incidents.csv`**.
   ZIP obsahuje váš API kľúč: uchovajte ho súkromne. ZIP aj `.env` sú ignorované Gitom.
4. Spustite **krok 1 — prostredie**. Rozbalí balík, nainštaluje knižnice, načíta API
   kľúč z `.env` bez vypísania hodnoty a pripojí Google Drive.
5. Spustite **krok 2 — tréning**. Pripraví dáta a EDA, potom postupne natrénuje všetky
   štyri modely. Počkajte na `TRENING DOKONCENY — 4/4 MODELY`.
6. Spustite **krok 3 — validácia**. Vytvorí BERT kandidátov, zavolá Gemini a vypíše
   validačné ST1/ST2 skóre. Modely sa už netrénujú.
7. Po kontrole validácie a uzamknutí nastavení spustite **krok 4 — finálny test**.
   Získate testovacie skóre a grafy do správy. Nepoužívajte Run All pred touto kontrolou.

Pri prerušení tréningu zopakujte kroky **1 → 2**. `--resume` obnoví checkpoint
vrátane optimalizátora; hotové modely sa preskočia. Po prerušení Gemini zopakujte
príslušný krok 3 alebo 4, ktorý využije cache hotových odpovedí.

Výstupy sa ukladajú do `/content/drive/MyDrive/semeval_task9/baseline_v1`.
Pripravte na Drive približne **15 GB voľného miesta** na štyri modely a obnoviteľné
checkpointy. Tréning používa lokálne dáta v pamäti; väčšie zápisy na Drive prebiehajú
pri checkpointoch. Zmena dát alebo tréningovej konfigurácie vyžaduje nový `RUN_DIR`.

### Čo robiť po dotrénovaní

Dokončený tréning znamená, že máte štyri uložené modely na výber kandidátov.
Na finálny baseline treba ešte dokončiť kroky **3 a 4**: modely vytvoria kandidátov,
Gemini z nich vyberie odpovede a evaluátor ich porovná s anotáciami.
Nie je potrebné trénovať znova, pokiaľ nemeníte tréningovú konfiguráciu alebo dáta.

V `MyDrive/semeval_task9/baseline_v1/predictions/` nájdete:

| Výstup | Použitie |
|---|---|
| `test.gemini.metrics.json` | Finálne ST1/ST2 skóre a metriky jednotlivých cieľov |
| `test.bert.metrics.json` | Kontrolné výsledky samotných klasifikátorov |
| `test.gemini.hazard-category.png`, `test.gemini.product-category.png` | Konfúzne matice do PDF |
| `test.candidate_report.json` | Candidate recall a priemerný počet kandidátov |
| `test.gemini.jsonl` a `test.gemini.jsonl.audit.jsonl` | Predikcie, prompty a odpovede pre analýzu chýb |

Do PDF doplňte výsledky, skutočné parametre z `models/<task>/run.json` a počty EDA
z `data/augmentation.json`. Porovnajte skóre s referenčnými systémami, opíšte chyby
a obmedzenie nevidených produktových tried. Uchovajte priečinok `models` aj výsledky
na Drive; po ukončení práce môžete Colab odpojiť.

## Oprava produktového klasifikátora

Ak `product` predikuje konštantne jeden produkt, použite
[`notebooks/repair_product_colab.ipynb`](notebooks/repair_product_colab.ipynb).
Má tri kroky: diagnostika pôvodného behu, tréning iba produktu a validačné overenie
bez Gemini. Nový model ukladá do `models/product_v2`; ostatné tri modely a pôvodný
produkt nemení. Podrobnosti a dôkazy sú v [`docs/product_repair.md`](docs/product_repair.md).

Opravná konfigurácia zachováva binárnu focal loss, ale vyvažuje pozitívne a negatívne
výstupy, vyhodnocuje celé epochy a vyberá checkpoint podľa recall@5. Pre tento beh
používajte `run_repair`, ktorý sa definuje v opravnom notebooku.

## Lokálne spustenie

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[test]"
.venv/Scripts/python -m nltk.downloader wordnet omw-1.4
.venv/Scripts/python -m baseline.cli prepare
.venv/Scripts/python -m baseline.cli augment
.venv/Scripts/python -m pytest -q
```

Pre lokálny GPU tréning treba príslušné CUDA/ROCm zostavenie PyTorchu. Na Colabe sa
použije už prítomný PyTorch s CUDA; CPU slúži na prípravu dát a automatické testy.
Nasledujúce príkazy spúšťajte v prostredí s dostupnou GPU:

```bash
python -m baseline.cli train --task hazard-category --resume
python -m baseline.cli train --task product-category --resume
python -m baseline.cli train --task hazard --resume
python -m baseline.cli train --task product --resume
python -m baseline.cli predict --split validation
python -m baseline.cli evaluate --split validation --predictions artifacts/baseline/predictions/validation.bert.jsonl
```

`--run-dir /cesta/k/vystupom` a `--config configs/baseline.json` sa zadávajú **pred**
podpríkazom. Na dvojkrokové overenie tréningu použite `train --task hazard-category --smoke`.
Smoke výsledky sú oddelené a nemožno ich použiť na finálnu predikciu.

## Gemini a finálna predikcia

Lokálne sa kľúč načíta z `.env`. Súbor je ignorovaný Gitom.
`python -m baseline.cli gemini-models` vypíše dostupné modely bez generovania textu.
Predvolený model bol overený reálnym generovaním. Starší 2.5 Flash-Lite sa síce
zobrazoval v zozname, ale generovanie pre tento účet odmietol. Používa sa
`thinking_level: minimal`. Pri zmene modelu overte podporu nastavenia uvažovania;
`thinking_budget` a `thinking_level` sa nesmú nastaviť súčasne a `null` ich vynechá.

Najprv vygenerujte prompty bez volania API, potom pilot na **25 vzorkách = 100 promptoch**:

```bash
python -m baseline.cli llm --candidates artifacts/baseline/predictions/validation.candidates.jsonl --output artifacts/baseline/predictions/validation.pilot.gemini.jsonl --limit 25 --dry-run
python -m baseline.cli llm --candidates artifacts/baseline/predictions/validation.candidates.jsonl --output artifacts/baseline/predictions/validation.pilot.gemini.jsonl --limit 25
python -m baseline.cli evaluate --split validation --predictions artifacts/baseline/predictions/validation.pilot.gemini.jsonl --allow-partial
```

Pre plnú validáciu použite rovnaký príkaz `llm` bez `--limit`, **s rovnakým výstupným
súborom ako pilot**, aby sa prvé odpovede znova použili z cache. Potom vyhodnoťte celý
súbor bez `--allow-partial`. Názov s `pilot` je len názov súboru; úplnosť overuje evaluátor.

Po uzamknutí nastavení spustite `predict --split test`, `llm` nad testovacími
kandidátmi a `evaluate --split test`. Test sa nepoužíva na tréning, early stopping ani
výber prahov. Samotná príprava dát z neho vytvorí iba audit počtov/neznámych tried.

Gemini má pevný horný limit 5 HTTP pokusov/minútu s odstupom aspoň 13 sekúnd.
Hodnota `requests_per_minute: 10` zostáva v pôvodnej konfigurácii kvôli kompatibilite
s existujúcou cache; runtime ju obmedzí na 5.
Obmedzenie platí aj pre HTTP pokusy opakované po chybe. Pri HTTP 429 čaká aspoň
60 sekúnd pred ďalším pokusom (alebo dlhšie podľa `Retry-After`). Na spomalenie
bez zneplatnenia existujúcej cache použite premennú prostredia
`GEMINI_REQUESTS_PER_MINUTE=5` (alebo nižšiu), nie zmenu existujúcej konfigurácie experimentu.
Každá vzorka používa štyri požiadavky (validácia 2 260, test 3 988). Pilot tvorí časť
validácie. Cache zohľadňuje celý prompt, model aj nastavenia; zmenený experiment
nevydáva staré odpovede za nové. API chyby zastavia beh a zachovajú hotové výsledky.
Neplatný alebo blokovaný textový výstup použije najlepšieho BERT kandidáta a označí
sa ako fallback. Číslované možnosti sú zoradené podľa skóre; odpoveď sa mapuje presne,
potom cez podobnosť `SequenceMatcher >= 0.9`, napokon cez BERT fallback.
Teplota 0 obmedzuje náhodnosť, úplnú deterministickosť služby nezaručuje.

## Metodika a predvolené parametre

- Iba anglické záznamy a pôvodný `semeval-split`: **5 082 / 565 / 997**.
- Vstup je vyčistený `title + text`, malé písmená, normalizované medzery a odstránené
  duplicitné celé riadky. Ostatné stĺpce nie sú modelovým vstupom.
- Jeden spoločný EDA dataset pre všetky modely; dopĺňa existujúce produktové triedy
  do 10 vzoriek. Každý variant používa jednu zo štyroch EDA operácií, ktoré sa striedajú
  pri tvorbe variantov; miera 0.1 sa počíta z upraviteľných slov. Táto konkrétna voľba
  spresňuje nejednoznačné znenie PDF o aplikovaní štyroch operácií na každý variant.
- EDA zachováva nadpis, negácie, čísla, slová všetkých štyroch labelov a všetky labely.
  Chránené viacslovné názvy sa pri zámene nerozdeľujú. Synonymá poskytuje WordNet.
  Ochrana je lexikálna, nie úplná sémantická záruka; vzorky treba manuálne pozrieť.
  Duplicitné varianty sa nepridávajú. Ak nemožno vytvoriť dosť unikátnych textov,
  nedostatok sa uvedie v `augmentation.json`; počet 11 985 sa nepredpokladá naslepo.
- Každý cieľ má samostatný enkodér a sigmoidovú hlavu s one-hot tréningovým cieľom.
  Binárna focal loss: **gamma 2, alpha 0.25**, priemer cez vzorky aj výstupy.
- AdamW, learning rate **1e-5**, weight decay 0.01, warmup 10 %, max. **5 epoch**,
  batch 8, akumulácia 2 (efektívne 16), max. **512 tokenov**, FP16 na GPU,
  gradient checkpointing. Predčasné zastavenie po 2 vyhodnoteniach bez zlepšenia
  validačného macro-F1; vyhodnotenie a checkpoint každých 250 aktualizácií.
- Kandidáti: sigmoid skóre **> 0.5**, pri prázdnej množine **top-5**, maximum **20**.
  Predvolene sa filtrujú aj ST1 kategórie, aby celý baseline používal rovnakú pipeline.
  `st1_all_labels: true` voliteľne zahrnie všetky tréningové kategórie pre ST1.
- LLM dostáva celý vyčistený text do 50 000 znakov (zdrojové texty sa do limitu zmestia),
  nie iba BERT výrez. Neposielajú sa gold labely ani pomocné anotačné stĺpce.
- Slovníky sú len z tréningu: 10/22 kategórií, 128 hazardov, **1 022 produktov**.
  V validácii je **49 vzoriek so 46 nevidenými produktmi**. Nevyraďujú sa; candidate
  recall sa uvádza celkovo aj pre známe triedy. EDA nevytvára nové produktové triedy.

## Výsledky a reprodukovateľnosť

Stiahnuté predikcie možno rýchlo vyhodnotiť lokálne bez GPU a bez Gemini API:

```powershell
.venv/Scripts/python scripts/print_metrics.py
```

Predvolene číta `evaluated_predictions/` a originálny dataset. Vypíše accuracy,
macro precision, macro recall, macro-F1, ST1/ST2, porovnanie BERT a Gemini na rovnakých dokončených vzorkách a recall
kandidátov. Iný priečinok zvoľte cez `--predictions-dir`, test cez `--split test`.
Voliteľne uložte JSON pomocou `--output artifacts/downloaded_metrics.json`.
Zdrojové predikcie nemení a pri každom porovnaní uvedie počet hodnotených vzoriek.
Oficiálne skóre ST1 a ST2 zobrazí aj v samostatnej porovnávacej tabuľke.
Publikované výsledky organizátorských baseline modelov a systémov Anastasia, MyMy,
SRCB a PATeam sú uvedené v samostatnej tabuľke zo správy Randl et al. (2025),
tabuľky 2 a 3 (`sources/SemEval_2025_Report.pdf`, PDF strana 5). Ide o publikované
testovacie výsledky iných systémov; nenahrádzajú naše namerané hodnoty.
Rozklad podľa rovníc 2–4 ukazuje macro-F1 hazardu, macro-F1 produktu na vzorkách
so správnym hazardom, počet týchto vzoriek a ich priemer ako výsledné skóre.
Accuracy, precision, recall a macro-F1 v úvodných tabuľkách sú doplnkové metriky
jednotlivých cieľov na všetkých uvedených vzorkách; produktová hodnota z tejto
tabuľky preto nemusí byť rovnaká ako podmienená produktová zložka skóre ST1/ST2.
Ak priečinok obsahuje aj `test.bert.jsonl` alebo `test.gemini.jsonl`, predvolený výpis
pridá samostatnú tabuľku skóre na testovacej množine. Bez týchto súborov zobrazí,
že testovacie skóre nie je dostupné. Iba test vypíšete cez `--split test`.

`data/manifest.json` obsahuje SHA-256 CSV, počty, triedy a audit rovnakých textov medzi
splitmi. `augmentation.json` obsahuje seed, operácie, reálne počty a odtlačok dát.
`models/<task>/run.json` zaznamenáva verzie knižníc, GPU, model revision a parametre.
`history.json` obsahuje priebeh tréningu. Checkpointy zahŕňajú aj stav optimalizátora.
Záznam balíkov v `environment.json` z notebooku umožňuje spätne zostaviť prostredie;
rozsahy verzií v `pyproject.toml` samy osebe nie sú úplný lockfile.

`predictions/*.candidates.jsonl` obsahuje kandidátov a skóre; `*.bert.jsonl` kontrolnú
predikciu bez LLM. Gemini `*.audit.jsonl` obsahuje prompty, surové odpovede, model,
mapovanie a tokenovú spotrebu, ale žiadne API kľúče. `*.metrics.json` obsahuje oficiálne
ST1/ST2 skóre, samostatné macro-F1 všetkých cieľov a per-class reporty. ST1 konfúzne
matice sú PNG. Neúplný pilot je vždy označený ako neúplný.

Oficiálna metrika priemeruje hazard macro-F1 a produkt macro-F1 **iba na vzorkách so
správnym hazardom**. Používa sa predvolená množina tried `sklearn` (zjednotenie gold a
predikcie danej hodnotenej množiny). Pri žiadnom správnom hazarde je produktová zložka
0, čo zodpovedá slovnému pravidlu organizátorov a odstraňuje neurčité NaN prázdnej množiny.

Použitie AI pri vývoji eviduje [`docs/ai_usage.md`](docs/ai_usage.md). Skutočné výsledky
do PDF doplňte až po plnom tréningu a vyhodnotení; smoke test nie je výsledok baseline.

## Zdroje

- [Oficiálny kód metriky](https://github.com/food-hazard-detection-semeval-2025/food-hazard-detection-semeval-2025.github.io#evaluation)
- [Oficiálna správa súťaže](https://aclanthology.org/2025.semeval-1.325/)
- [EDA — Wei a Zou, 2019](https://aclanthology.org/D19-1670/)
- [Focal loss — Lin a kol., 2017](https://openaccess.thecvf.com/content_ICCV_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)
- [Gemini generateContent API](https://ai.google.dev/api/generate-content)
