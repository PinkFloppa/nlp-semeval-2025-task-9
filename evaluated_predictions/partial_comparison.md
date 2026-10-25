# Vyhodnotenie validacie

Dataset: 565 vzoriek. BERT: 565 predikcii. BERT + Gemini: 122 predikcii.

## Oficialne publikovane vysledky - testovacia mnozina

| System z clanku | ST1 | ST2 |
|---|---:|---:|
| TF-IDF + LR (baseline organizatorov) | 0.4980 | 0.1830 |
| BERT (baseline organizatorov) | 0.6670 | 0.1650 |
| Anastasia | 0.8223 | 0.1281 |
| MyMy | 0.8112 | 0.5278 |
| SRCB | 0.8039 | 0.5473 |
| PATeam | 0.8017 | 0.5266 |

Zdroj: [Randl et al. (2025): SemEval-2025 Task 9: The Food Hazard Detection Challenge](https://aclanthology.org/2025.semeval-1.325/), tabulky 2 a 3; lokalny subor `sources/SemEval_2025_Report.pdf`, PDF strana 5.

Ide o publikovane hodnoty inych systemov na teste. Nase namerane hodnoty su uvedene samostatne nizsie.

## BERT - vsetky dostupne predikcie (565 vzoriek)

| Ciel | Spravne | Accuracy | Precision (macro) | Recall (macro) | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| hazard-category | 540/565 | 95.58% | 87.52% | 86.62% | 0.8642 |
| product-category | 438/565 | 77.52% | 71.97% | 65.66% | 0.6724 |
| hazard | 419/565 | 74.16% | 19.06% | 22.93% | 0.2051 |
| product | 0/565 | 0.00% | 0.00% | 0.00% | 0.0000 |

Nase namerane skore: **ST1 = 0.7897**, **ST2 = 0.1025**.

## BERT - rovnake vzorky ako Gemini (122 vzoriek)

| Ciel | Spravne | Accuracy | Precision (macro) | Recall (macro) | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| hazard-category | 114/122 | 93.44% | 84.44% | 81.54% | 0.8248 |
| product-category | 107/122 | 87.70% | 80.03% | 79.43% | 0.7872 |
| hazard | 81/122 | 66.39% | 20.03% | 25.93% | 0.2177 |
| product | 0/122 | 0.00% | 0.00% | 0.00% | 0.0000 |

Nase namerane skore: **ST1 = 0.8016**, **ST2 = 0.1089**.

## BERT + Gemini - rovnake vzorky (122 vzoriek)

| Ciel | Spravne | Accuracy | Precision (macro) | Recall (macro) | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| hazard-category | 114/122 | 93.44% | 88.15% | 81.54% | 0.8248 |
| product-category | 108/122 | 88.52% | 80.50% | 83.93% | 0.8096 |
| hazard | 90/122 | 73.77% | 35.03% | 39.27% | 0.3571 |
| product | 0/122 | 0.00% | 0.00% | 0.00% | 0.0000 |

Nase namerane skore: **ST1 = 0.8133**, **ST2 = 0.1786**.

## Zlozky nasho skore

Macro-F1 hazardu sa pocita zo vsetkych vzoriek. Produkt sa hodnoti iba tam, kde je hazard spravny. Skore je priemer tychto dvoch macro-F1. Pre ST1 sa pouzivaju kategorie, pre ST2 konkretne oznacenia.

| Model | Uloha | F1 hazardu | F1 produktu pri spravnom hazarde | Spravny hazard | Skore |
|---|---|---:|---:|---:|---:|
| BERT - vsetky vzorky | ST1 | 0.8642 | 0.7151 | 540/565 | 0.7897 |
| BERT - vsetky vzorky | ST2 | 0.2051 | 0.0000 | 419/565 | 0.1025 |
| BERT - rovnake vzorky | ST1 | 0.8248 | 0.7785 | 114/122 | 0.8016 |
| BERT - rovnake vzorky | ST2 | 0.2177 | 0.0000 | 81/122 | 0.1089 |
| BERT + Gemini | ST1 | 0.8248 | 0.8017 | 114/122 | 0.8133 |
| BERT + Gemini | ST2 | 0.3571 | 0.0000 | 90/122 | 0.1786 |

Tabulky jednotlivych cielov vyssie obsahuju doplnkove metriky na vsetkych uvedenych vzorkach. Produktove macro-F1 v nich nie je podmienena produktova zlozka oficialneho skore.

## Recall kandidatov
Pocet vzoriek: 565.

| Ciel | Recall kandidatov |
|---|---:|
| hazard-category | 96.28% |
| product-category | 87.79% |
| hazard | 81.59% |
| product | 0.00% |

## Kontrola produktoveho modelu
BERT predikuje na vsetkych 565 vzorkach ground turmeric. Spravny produkt sa nedostal medzi kandidatov; recall kandidatov je 0 %.

Pocet zachovanych Gemini odpovedi v audite: 491.
