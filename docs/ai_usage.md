# Použitie AI pri implementácii baseline

Dátum: 25. 10. 2026. Nástroj: Codex (asistent v tomto pracovnom priestore).
Táto evidencia dopĺňa, nenahrádza existujúcu evidenciu ostatných nástrojov v správe.

## Účel a zadanie

Používateľ požiadal o prečítanie `Projektove_zadanie.pdf` a
`SemEval_2025_Task_9_1_main_no_parameters.pdf`, návrh postupu a následne implementáciu.
Finálny implementačný prompt: „Dobre podme implementovat baseline riešenie podľa nášho pdf“.
Predchádzajúce spresnenia: dodržať EDA, štyri one-vs-all BERT modely s binárnou focal
loss a LLM bez fine-tuningu; použiť Gemini API; tréning na bezplatnom Colab kerneli
`SEMEVAL_T4`. Overovací notebook potvrdil Tesla T4 a funkčnú CUDA.

## Relevantné výstupy

Asistent vytvoril Python balík `baseline`, konfiguráciu, testy, návod, skript na
bezpečné balenie zdrojov a notebook pre Colab. Vývoj zahŕňal kontrolu dátových splitov,
identifikáciu nevidených produktových tried, implementáciu EDA, focal loss, tréningu,
výberu kandidátov, Gemini REST klienta a oficiálnej hodnotiacej rovnice.

Celá konverzácia obsahuje aj vysvetlenia EDA, úloh BERT/LLM a orientačné odhady času.
Tieto odhady nie sú namerané výsledky. Pre úplnú prílohu uchovajte/exportujte celú
konverzáciu vrátane relevantných promptov a odpovedí; tento súbor je stručný register.

Prompty a odpovede Gemini použité **vo výslednom riešení** sa automaticky ukladajú
do `*.audit.jsonl` vrátane konkrétnej verzie modelu, ak ju API vráti, a tokenovej spotreby.
API kľúče sa do evidencie neukladajú. Na výslovnú žiadosť používateľa bol prenosný
ZIP upravený tak, aby obsahoval `.env` aj dataset. Tento súkromný ZIP a `.env` sú
vylúčené z Gitu. Notebook bol zjednodušený na štyri očíslované kroky:
prostredie, tréning, validácia, finálny test. Kľúč načítava bez vypísania hodnoty.

Stav skutočných kontrol a meraní: pozri `docs/verification.md`.

Následný používateľský prompt požadoval diagnostiku konštantnej produktovej predikcie
a opravu s opätovným tréningom v Colabe. Výstupom je `docs/product_repair.md`,
vyvážená redukcia binárnej focal loss a samostatný opravný notebook. Diagnostika
rozlišuje pozorované zlyhanie, kontrolovaný gradientový výpočet a zatiaľ nenameraný
výsledok plného opraveného tréningu. Ostatné tri modely ostávajú zachované.
