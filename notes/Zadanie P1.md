## 1. Formulácia problému, motivácia, ciele projektu

- Model má pomocou features : year, month, day, country a title identifikovať najprv širšiu kategóriu hazardu ktorý sa nachádza v produkte a akej kategórie produktu sa to týka, potom má naďalej upresniť špecifický typ hazardu a produktu

- V datasete je taktiež atribút text - toto je plné telo textu ktorého nadpisom je title

- Pre kat. hazardov existuje 10 labelov, kat. produktu - 22 labelov
- Pre špecifické hazardy existuje 128 labelov a špecifické produkty 1 142 labelov

- Motivácia : Ochrana zdravia ľudí pred hazardmi ktoré sa objavia v jedle. Drúhá časť klasifikácie taktiež slúži na vysvetlenie a podporu prvej časti (kategórie hazardu a produktu). Táto oblasť vysvetlivosti je inak nedostačne preskúmaná 

## 2. Prehľad relevantnej literatúry, dátových zdrojov a existujúcich riešení

- Zdroj dát : 24 verejný stránok autorít týkajúcich sa bezpečnosti jedla pomocou platformy AgroKnow

Report súťaže : https://aclanthology.org/2025.semeval-1.325.pdf

V rámci súťaže boli publikované tri rôzne baseliny :

**Baseline 1: TF-IDF + logistická regresia**

1. vstup = titulok ->
2. vektorizácia textu pomocou TF-IDF (scikit-learn) ->
3. tréning logistickej regresie ako multi-class klasifikátora pre daný label (hazard-category, product-category, hazard, product) ->
4. predikcia labelu

**Výsledok:** ST1 = 0.498, ST2 = 0.183

**Baseline 2: BERT**

1. vstup = titulok ->
2. tokenizácia (bert-base-uncased) ->
3. fine-tuning `bert-base-uncased` ako multi-class klasifikátora (knižnica huggingface transformers) ->
4. predikcia labelu

**Výsledok:** ST1 = 0.667, ST2 = 0.165 (v ST2 horší ako TF-IDF + LR)

**Baseline 3: CICLe (Conformal In-Context Learning)**

1. vstup = text reportu ->
2. natrénovanie jednoduchého základného klasifikátora ->
3. konformná predikcia (knižnica crepes): z výstupov klasifikátora sa vytvorí malá množina pravdepodobných labelov namiesto jedného ->
4. prompt pre veľký LLM (napr. GPT-4, bez fine-tuningu) s touto zúženou množinou kandidátov, prípadne aj s príkladmi ->
5. LLM vyberie finálny label