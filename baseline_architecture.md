# Baseline Recipe: SemEval-2025 Task 9 (Food Hazard Detection)

A simplified SRCB-style pipeline: EDA data augmentation, BERT candidate recall with focal loss, and zero-shot LLM prediction with a multiple-choice prompt.

## 1. Data preparation

- Use the official splits: train (5,082), validation (565), test (997). Keep the validation set untouched for evaluation.
- Concatenate **title + text** into one input text.
- Clean the text as SRCB did: normalise spaces and line breaks, remove duplicate segments, lowercase.
- Treat the four label types as **four separate classification problems**: hazard-category and product-category (ST1), hazard and product (ST2).

## 2. Data augmentation with EDA (training set only)

- Augment **after** splitting and **only** the training set, so no synthetic data ends up in validation.
- Create **one shared augmented dataset** for all four classifiers: each augmented sample keeps **all four labels** of the sample it was created from.
- Drive augmentation by the **exact product** label, since it has the most rare classes; this also boosts the other labels.
- Fill each rare class up to **10 samples** by creating EDA variants of its existing samples.
- Apply the four EDA operations (Wei and Zou, 2019): **synonym replacement** (WordNet), **random insertion**, **random swap** and **random deletion**.
- Change only a small fraction of words per variant, e.g. **10%** (the original EDA paper's recommended setting).
- **Protect the label words:** don't modify the hazard and product terms (e.g. "listeria monocytogenes", "smoked sausage"), and leave the **title** unchanged; apply EDA only to the text.
- Use an existing implementation, e.g. `nlpaug` or the original EDA script, and **fix the random seed**.
- Spot-check a few augmented samples by hand.
- Record the operations, the change rate, the target count per class and the seed.

## 3. Candidate recall module (BERT)

- Choose one encoder: **DeBERTaV3-base** (slightly better recall in SRCB's experiments) or **RoBERTa-base**.
- Train **one model per classification problem** (4 in total) on a free **Colab or Kaggle GPU**; split the runs across teammates if needed.
- Use a **one-vs-all** setup: one sigmoid output per label, i.e. a multi-label head on a shared encoder.
- Loss: **binary (sigmoid) focal loss**, following Anastasia, e.g. `torchvision.ops.sigmoid_focal_loss`. Start with γ = 2 and α = 0.25, then tune.
- Hyperparameters to start from (SRCB): max length 512, learning rate 1e-5, AdamW, early stopping on validation.
- Select candidates as all labels with **probability > 0.5**, with two safeguards:
  - if no label passes, take the **top-k** labels (e.g. k = 5);
  - cap the list at a maximum (e.g. 10–20) to keep the prompt short.

## 4. LLM prediction (zero-shot, no fine-tuning)

- Build a **multiple-choice prompt** per sample containing:
  - a short task instruction,
  - the cleaned title and text,
  - the candidate labels as a numbered list,
  - an instruction to answer with **exactly one** label from the list and nothing else.
- For **ST1**, filtering is optional: with only 10 hazard categories and 22 product categories, you can list **all** labels.
- Use **temperature 0** for deterministic outputs, and turn off any "thinking" or reasoning mode.
- Order the candidates by their BERT probability, highest first, and keep that order fixed.
- Parse the answer: exact match to a label, otherwise the closest match (e.g. fuzzy string matching), otherwise fall back to the top BERT candidate.
- **Pilot first:** run ~100 prompts to test the prompt and the parsing before the full run (~6,000 prompts).

Model recommendations (check current versions and prices before choosing):

- **API, no GPU needed:** Gemini 2.5 Flash-Lite or Gemini 3.1 Flash-Lite (cheapest), GPT-5.4 mini, Claude Haiku.
- **Local, free Colab/Kaggle GPU:** Qwen2.5-7B-Instruct, Qwen3-8B, Llama-3.1-8B-Instruct.

## 5. Evaluation

- Compute the **official task score** for each sub-task: hazard macro-F1, product macro-F1 only on samples with a correct hazard, averaged. Ideally use the organisers' evaluation script.

## Estimated cost

The API cost comes only from the prediction step: roughly **$1–6** depending on the model, plus margin for pilot runs and reruns.
