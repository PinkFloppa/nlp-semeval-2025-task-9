# Diagnostika a oprava produktového modelu

## Pozorované zlyhanie

Zo stiahnutých validačných predikcií: všetkých 565 top-1 produktov je
`ground turmeric`, presnosť produktu a recall kandidátov sú 0. Platí to aj pre
produkty známe z tréningu; samotných 49 nevidených produktových vzoriek to nevysvetľuje.
Kandidáti majú takmer rovnaké poradie nezávisle od textu.

V `validation.sources.json` je produktový tréning ukončený po 750 aktualizáciách
(1 epoche). Ako najlepší bol vybraný checkpoint 250. Pôvodná patience 2 počíta
vyhodnotenia každých 250 aktualizácií, nie celé epochy; prvá kontrola bola navyše
ešte v 10 % warmupe približne 375 aktualizácií päťepochového plánu.

Pôvodné váhy a plná história nie sú v lokálne stiahnutých súboroch. Nemožno teda
bez ďalších dát potvrdiť jedinú príčinu (napr. vylúčiť FP16 overflow). Opravný notebook
na Drive načíta aj pôvodnú históriu vrátane nekonečných/NaN hodnôt `grad_norm`, ak je dostupná.

## Reprodukovaný problém redukcie straty

Pôvodná loss je korektná binárna focal loss priemerovaná cez všetky výstupy.
Pre produkt je však 1 pozitívny oproti 1 021 negatívnym výstupom. Pri nulových
logits a `alpha=0.25` je pomer súhrnných absolútnych negatívnych logitových gradientov
k pozitívnemu `3 * 1021 = 3063`. Ide o kontrolovaný výpočet v počiatočnom bode,
nie o meranie gradientov pôvodného enkodéra počas celého tréningu.

Focal modulácia pri rovnako neistých výstupoch tento pomer sama neodstráni. Model
môže znižovať loss hlavne potláčaním všetkých výstupov a predčasná kontrola macro-F1
zafixuje slabý skorý checkpoint. Znížená loss preto sama nedokazuje naučenú klasifikáciu.

## Zmena iba pre produkt

Opravná konfigurácia `configs/product_repair.json` zachováva RoBERTa/BERT rodinu,
EDA, štyri nezávislé modely a sigmoidovú binárnu focal loss. Pre produkt mení:

| Parameter | Pôvodný | Oprava |
|---|---|---|
| Redukcia focal loss | priemer všetkých výstupov | priemer pozitívnych a negatívnych osobitne |
| Alpha | 0.25 | 0.5 |
| Learning rate | 1e-5 | 2e-5 |
| Vyhodnotenie/checkpoint | každých 250 aktualizácií | každá celá epocha |
| Výber checkpointu | macro-F1 top-1 | recall@5 |
| Patience | 2 vyhodnotenia | 3 celé epochové vyhodnotenia |
| Maximum epoch | 5 | 5 |

Nová strata je `alpha * mean(positive focal) + (1-alpha) * mean(negative focal)`.
Pri nulových logits má s alpha 0.5 pomer vyššie uvedených gradientových súčtov 1:1.
Záporné príklady sa stále učia; nejde o softmax alebo zmenu na multiclass cross-entropy.
Tréningové frekvencie sa neprenastavujú podľa validačných či testovacích odpovedí.

Recall@5 zodpovedá účelu výberu kandidátov, ale nie je finálne súťažné skóre.
Oficiálne ST1/ST2 vyhodnotenie sa nemení. Nevidené triedy zostávajú vo validácii
ako nesprávne predikcie; nezískavame ich z testovacích anotácií.

## Nasadenie do Colabu

1. Nahrať aktuálny `baseline_colab.zip` a reštartovať kernel `SEMEVAL_T4`.
2. Otvoriť `notebooks/repair_product_colab.ipynb`, nie pôvodný notebook pre všetky modely.
3. Spustiť krok 1 (prostredie a diagnostika), potom 2 (iba nový produktový tréning).
4. Spustiť krok 3: kandidáti a metriky na validácii bez Gemini volaní.

Ostatné tri modely sa načítajú z pôvodných ciest. Starý produkt ostáva v
`models/product`, nový je v `models/product_v2`. Výstupy sú oddelené v
`predictions/product_v2`; žiadne pôvodné metriky či API odpovede sa neprepisujú.
`run_repair` vyberá opravnú konfiguráciu; pôvodné `run` stále používa pôvodnú.

Kontrola po tréningu uloží `validation_health.json`: konečné logits musia byť
konečné čísla, predikcie nesmú byť konštantné, accuracy, recall@5 aj skutočný recall
kandidátov musia byť nenulové. Pri zlyhaní vznikne `failed.json`, nie `complete.json`.
To je základná kontrola zlyhania, nie záruka vysokej kvality modelu.

Nový priečinok predikcií dostane kópiu starej Gemini cache. Zhodné prompty troch
nezmenených cieľov môžu odpovede znovu použiť; nové produktové prompty potrebujú nové
volania. Rýchlosť API ani konfigurácia LLM sa týmto experimentom nemenia.

## Overenie

Automatické testy kontrolujú pomer gradientov 3063:1 → 1:1, stabilitu pri extrémnych
logits, nezávislosť redukcie od počtu rovnakých negatívnych výstupov, kontrolu kolapsu,
izoláciu konfigurácie a skutočný krátky tréning malého BERT modelu s 1 022 výstupmi.
Samostatná 1 022-výstupová lineárna hlava sa na kontrolovanej osemtriedovej úlohe
naučí správne triediť; tento syntetický test nenahrádza meranie na potravinovom datasete.

**Plný nový tréning na T4 a skutočné zlepšenie validačného produktu sa musia ešte
zmerať v Colabe.** Doterajšie pôvodné výsledky sa nevydávajú za výsledky opravy.

Referencie k použitej implementácii:
[PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html),
[Hugging Face EarlyStoppingCallback](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/callback#transformers.EarlyStoppingCallback).
