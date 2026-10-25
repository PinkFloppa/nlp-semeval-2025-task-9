# Overenie implementácie — 25. 10. 2026

## Vykonané kontroly

- Automatické testy: **12 úspešných** (`.venv/Scripts/python -m pytest -q`).
- Porovnanie hodnotiacej funkcie s oficiálnou referenčnou rovnicou/kódom organizátorov
  na syntetických predikciách; samostatne overený nulový počet správnych hazardov.
- Oficiálne anglické splity: **5 082 tréning, 565 validácia, 997 test**.
- Slovníky len z tréningu: **10, 22, 128, 1 022 tried** pre štyri ciele.
- Plná EDA: **6 903** syntetických + **5 082** pôvodných = **11 985** záznamov.
  Všetky existujúce produktové triedy dosiahli aspoň 10 vzoriek.
- EDA operácie: 2 176 synonym replacement, 1 664 random insertion,
  1 587 random swap, 1 476 random deletion. Každý variant používa jednu operáciu.
- Opakovaná EDA s rovnakým seedom vytvorila rovnaký odtlačok:
  `468b991f4c85b3eab0dafa297359b69af59425e3799066762db789e67a84bbda`.
- Kontrola všetkých syntetických záznamov: rodič výhradne z tréningu, nezmenený
  nadpis aj všetky štyri labely, neodstránené slová labelov a chránené negácie.
- Skutočný dvojkrokový CPU tréning malého náhodne inicializovaného BERT modelu:
  dopredný/spätný priechod, focal loss, validácia, checkpoint, optimizer state,
  načítanie výsledného modelu a obnovenie tréningu z checkpointu.
- Integračná predikcia štyroch malých klasifikátorov, výber kandidátov a evaluátor
  vrátane uloženia konfúznych matíc. Výstupy predikcie neobsahujú gold labely.
- API cache: opakovaný beh nevytvára nové požiadavky; posledný neúplný záznam po
  prerušení sa bezpečne odstráni, vnútorné poškodenie logu sa neignoruje.
- Reálne Gemini API: jednoduché spojovacie volanie a jedna klasifikačná ukážka na
  tréningovom zázname s ručne určenými kandidátmi. Model **gemini-3.5-flash-lite**,
  klasifikačná odpoveď `listeria monocytogenes`, presná zhoda, 231 vstupných a 5
  výstupných tokenov. Táto kontrola nie je skóre baseline.
- Starší 2.5 Flash-Lite odmietol generovanie HTTP 404 napriek prítomnosti v zozname
  modelov; predvolený model bol preto zmenený na overený 3.5 Flash-Lite.
- Overené načítanie tokenizéra a konfigurácie predtrénovanej RoBERTa z pripnutej
  revízie `e2da8e2f811d1448a5b465c236feacd80ffbac7b`.
- Skontrolovaná syntax všetkých kódových buniek notebooku a ignorovanie `.env` Gitom.

Testovacie prostredie: Windows, Python 3.13, PyTorch 2.14.1+cpu,
Transformers 4.57.6. Detaily sú uložené v ignorovaných lokálnych artefaktoch.
Predchádzajúci používateľom spustený GPU notebook potvrdil na Colabe Tesla T4,
14.56 GiB VRAM, Python 3.12.13, PyTorch 2.11.0+cu128 a funkčnú CUDA.

## Čo ešte nie je namerané

**Plný tréning RoBERTa na T4, kandidáti z týchto natrénovaných modelov a výsledné
ST1/ST2 skóre ešte neboli vykonané.** Vyžadujú spustenie pripraveného Colab notebooku
v používateľovej vzdialenej relácii, ku ktorej tento asistent nemá nástroj na vykonávanie
buniek. Notebook obsahuje krátku GPU kontrolu aj plné tréningové a vyhodnocovacie kroky.

Skóre sa nesmie odvodzovať z malých testovacích modelov ani jednej API ukážky.
Po plnom behu doplňte reálne výsledky, candidate recall, počet fallbackov a časy do správy.

## Následná oprava produktu

Po získaní validačných predikcií bol potvrdený kolaps produktu na `ground turmeric`
vo všetkých 565 vzorkách. Oprava a jej dôkazy sú v `docs/product_repair.md`.
Aktuálna kompletná sada má **30 úspešných testov**, vrátane vyváženej binárnej focal
loss, izolovaného nastavenia iba produktu, krátkeho BERT tréningu s 1 022 výstupmi,
kontroly konštantných predikcií a zachovania starého modelu. Kódové bunky oboch
Colab notebookov prešli syntaktickou kontrolou. Nové reálne validačné skóre produktu
sa ešte musí získať plným tréningom v opravnom Colab notebooku.
