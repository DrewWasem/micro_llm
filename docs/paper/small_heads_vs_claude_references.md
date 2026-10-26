# The 24 references in the small-heads paper, with exact titles


**13 of the paper's 18 related studies find a trained model matching or beating a prompted LLM.**


- **Titles are exact:** each was checked on 2026-10-26, 20 against arXiv and the other 4 against OpenAlex.
- **Summaries rest on abstracts and notes:** each comes from the source's abstract and the project's research notes, not a full read.
- **Three spots differ from the paper:** two title styles (entries 3 and 5) and one understated finding (entry 19).

## The data: one corpus of role-played support chats

### 1. Chen et al. (2021): the ABCD corpus every result comes from

- **Exact title:** Action-Based Conversations Dataset: A Corpus for Building More In-Depth Task-Oriented Dialogue Systems
- **Published in:** NAACL-HLT 2021, pages 3002–3017; arXiv:2104.00783.
- **What it did:** collected over 10,000 human-to-human support chats with 55 intents, each solved by a policy-bound sequence of agent actions.
- **Key result:** larger pretrained models beat simpler ones, yet trail human performance by 50.8 points of accuracy.
- **How the paper uses it:** the source of every chat and of both native labels (Sections 1, 2, 3.1 and 6).

## Seven studies find fine-tuned models beat prompted LLMs at classification

### 2. Ziems et al. (2024): 13 zero-shot LLMs fall short of fine-tuned models on 25 benchmarks

- **Exact title:** Can Large Language Models Transform Computational Social Science?
- **Published in:** *Computational Linguistics* 50(1), 2024; arXiv:2305.03514.
- **What it did:** tested 13 language models zero-shot on 25 English computational social science benchmarks.
- **Key result:** on classification the LLMs did not beat the best fine-tuned models, though they agreed fairly well with humans.
- **How the paper uses it:** broad evidence that prompted LLMs trail fine-tuned models (Section 2).

### 3. Kocoń et al. (2023): ChatGPT trails the state of the art by about 25% across 25 tasks

- **Exact title:** ChatGPT: Jack of all trades, master of none
- **Published in:** *Information Fusion*, article 101861, 2023; arXiv:2302.10724.
- **What it did:** ran ChatGPT on 25 analytical NLP tasks and analyzed over 49,000 responses, plus GPT-4 on five task subsets.
- **Key result:** ChatGPT averaged about 25% below the state of the art, and its loss grew with task difficulty; GPT-4's loss was lower.
- **How the paper uses it:** broad evidence that prompted LLMs trail task-specific models (Section 2).
- **The paper's version:** "ChatGPT: Jack of All Trades, Master of None", in title case. The journal uses sentence case, as above.

### 4. Edwards and Camacho-Collados (2024): fine-tuned small models still win on 16 datasets

- **Exact title:** Language Models for Text Classification: Is In-Context Learning Enough?
- **Published in:** LREC-COLING 2024; arXiv:2403.17661.
- **What it did:** compared zero- and few-shot LLMs with fine-tuned smaller models on 16 binary, multiclass and multilabel datasets.
- **Key result:** fine-tuning smaller, more efficient models still generally beat few-shot LLMs.
- **How the paper uses it:** broad evidence on text classification (Section 2).

### 5. Vajjala and Shimangaud (2025): zero-shot LLMs compete only on sentiment across 32 datasets

- **Exact title:** Text Classification in the LLM Era -- Where do we stand?
- **Published in:** arXiv:2502.11830.
- **What it did:** compared zero-shot, few-shot fine-tuned, synthetic-data and fully supervised classifiers on 32 datasets in 8 languages.
- **Key result:** zero-shot did well only on sentiment, and synthetic data from several LLMs built better classifiers than zero-shot open LLMs.
- **How the paper uses it:** broad evidence across languages (Section 2).
- **The paper's version:** "Text Classification in the LLM Era: Where Do We Stand?", with a colon for the double hyphen and title case.

### 6. Liu et al. (2022): few-shot fine-tuning beats in-context learning on accuracy and cost

- **Exact title:** Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning
- **Published in:** NeurIPS 2022 (*Advances in Neural Information Processing Systems* 35); arXiv:2205.05638.
- **What it did:** compared few-shot in-context learning with parameter-efficient fine-tuning, and introduced the (IA)³ method and the T-Few recipe.
- **Key result:** tuning a small set of parameters gave better accuracy at far lower compute, and T-Few beat human performance on the RAFT benchmark.
- **How the paper uses it:** evidence that light fine-tuning beats prompting on cost as well as accuracy (Section 2).

### 7. Bucher and Martini (2024): fine-tuned BERT-style models beat zero-shot GPT-4 and Claude Opus in every case

- **Exact title:** Fine-Tuned 'Small' LLMs (Still) Significantly Outperform Zero-Shot Generative AI Models in Text Classification
- **Published in:** arXiv:2406.08660.
- **What it did:** compared zero-shot GPT-3.5, GPT-4 and Claude Opus with fine-tuned BERT-style models on sentiment, approval, emotion and party-position tasks.
- **Key result:** fine-tuning won in every case, for example stance accuracy of 0.94 against 0.61 for Claude 3 Opus.
- **How the paper uses it:** one of two Claude comparisons among the broad studies (Section 2).

### 8. Dominguez-Olmedo et al. (2025): small fine-tuned models beat Claude 3.7 Sonnet on 260 legal tasks

- **Exact title:** Lawma: The Power of Specialization for Legal Annotation
- **Published in:** ICLR 2025; arXiv:2407.16615.
- **What it did:** built CaselawQA, 260 legal annotation tasks, and compared lightly fine-tuned small models with GPT-4.5 and Claude 3.7 Sonnet.
- **Key result:** a 135M-parameter model averaged 82% against 78% for Claude 3.7 Sonnet, and a few hundred to a thousand labels usually sufficed.
- **How the paper uses it:** evidence that a few hundred to a thousand labels can beat Claude on narrow tasks (Section 2).

## Intent detection: SetFit and prompted LLMs each win once in the two closest studies

### 9. Loukas et al. (2023): on 77 banking intents, SetFit edges the best LLM setting

- **Exact title:** Making LLMs Worth Every Penny: Resource-Limited Text Classification in Banking
- **Published in:** ICAIF '23 (ACM International Conference on AI in Finance); arXiv:2311.06102.
- **What it did:** priced few-shot LLMs from OpenAI, Cohere and Anthropic against SetFit and full fine-tuning on Banking77.
- **Key result:** SetFit with 10 examples per class scored 88.1 micro-F1 against 87.7 for the best LLM setting, and full fine-tuning reached 94.1.
- **How the paper uses it:** the closest intent-detection comparison with dollar costs (Section 2).

### 10. Arora et al. (2024): with retrieved examples, Claude 3 Haiku beats SetFit on intent detection

- **Exact title:** Intent Detection in the Age of LLMs
- **Published in:** EMNLP 2024 Industry Track; arXiv:2410.01627.
- **What it did:** compared 7 LLMs, prompted with retrieved examples and step-by-step reasoning, with SetFit on 6 intent datasets.
- **Key result:** Claude 3 Haiku averaged 0.736 F1 against 0.600 for SetFit, and routing only uncertain queries to the LLM kept accuracy within 2% at about 50% less latency.
- **How the paper uses it:** the contrast for Claude's missing examples (Section 2) and the untested router (Section 5.5).

### 11. Tunstall et al. (2022): SetFit, few-shot fine-tuning without prompts

- **Exact title:** Efficient Few-Shot Learning Without Prompts
- **Published in:** arXiv:2209.11055.
- **What it did:** introduced SetFit, which fine-tunes a sentence transformer on a few text pairs and trains a classifier on its embeddings.
- **Key result:** SetFit matched prompt-based few-shot methods with far fewer parameters and trained an order of magnitude faster.
- **How the paper uses it:** the few-shot method in the intent-detection studies (Section 2), and an untested alternative to the heads (Section 3.3).

## Three studies time or price both systems, and the small model is always faster

### 12. Lee et al. (2026): fine-tuned RoBERTa beats Claude Haiku 4.5 and answers in milliseconds

- **Exact title:** Long Live Fine-Tuning: Task-Specific Transformers Outperform Zero-Shot LLMs for Misinformation Response Classification on Reddit
- **Published in:** *Social Network Analysis and Mining*, 2026; arXiv:2606.04274.
- **What it did:** classified 900 Reddit replies to misinformation with nine models, including Claude Haiku 4.5, Claude Sonnet 4.6 and fine-tuned RoBERTa.
- **Key result:** RoBERTa reached 0.62 macro-F1 against 0.50 for the best zero-shot model, Claude Haiku 4.5, at 0.007 s per item against 10.48 s.
- **How the paper uses it:** a Claude Haiku comparison with time per item (Section 2).

### 13. Valdes Gonzalez (2026): fine-tuned encoders match or beat LLM prompting at a fraction of the cost

- **Exact title:** Cost-Aware Model Selection for Text Classification: Multi-Objective Trade-offs Between Fine-Tuned Encoders and LLM Prompting in Production
- **Published in:** arXiv:2602.06370.
- **What it did:** weighed the accuracy, latency and cost of fine-tuned encoders against zero- and few-shot prompting on IMDB, SST-2, AG News and DBPedia.
- **Key result:** encoders matched or beat prompting at one to two orders of magnitude lower cost and latency. Few-shot Claude Sonnet 4.5 still led on IMDB, 96.48 to 94.84.
- **How the paper uses it:** a cost and latency comparison in which each side wins one benchmark (Section 2).

### 14. Tran et al. (2025): a 1.5B-parameter router edges Claude 3.7 Sonnet at 51 ms against 1,450

- **Exact title:** Arch-Router: Aligning LLM Routing with Human Preferences
- **Published in:** arXiv:2506.16655.
- **What it did:** trained a 1.5B-parameter model that routes each query by matching it to user-defined domains and action types.
- **Key result:** it scored 93.17 against 92.79 for Claude 3.7 Sonnet, at 51 ms against 1,450 ms.
- **How the paper uses it:** a small model that matches Claude at a fraction of its latency (Section 2).

## Three studies use a large model to train or spare a smaller one

### 15. Hsieh et al. (2023): a 770M model distilled with LLM rationales beats 540B PaLM

- **Exact title:** Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes
- **Published in:** Findings of ACL 2023; arXiv:2305.02301.
- **What it did:** trained small models on labels plus rationales extracted from an LLM, in a multi-task setup.
- **Key result:** a 770M-parameter T5 beat few-shot PaLM 540B on ANLI with 80% of the training data, which standard fine-tuning failed to match with all of it.
- **How the paper uses it:** a cheaper path that keeps real inputs, unlike Section 5.3's synthetic tickets (Section 2).

### 16. Chen, Zaharia and Zou (2023): FrugalGPT matches GPT-4 at up to 98% lower cost

- **Exact title:** FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance
- **Published in:** arXiv:2305.05176. The paper cites arXiv only, because lookups could not confirm a venue.
- **What it did:** learned a cascade that decides which LLM APIs to query for each input.
- **Key result:** the cascade matched GPT-4 at up to 98% lower cost, or beat its accuracy by 4% at the same cost.
- **How the paper uses it:** a way to spend less on the large model while keeping real inputs (Section 2).

### 17. Li et al. (2023): LLM-written training data helps less as tasks grow more subjective

- **Exact title:** Synthetic Data Generation with Large Language Models for Text Classification: Potential and Limitations
- **Published in:** EMNLP 2023; arXiv:2310.07849.
- **What it did:** trained classifiers on LLM-written data and measured how each task's subjectivity changed their accuracy.
- **Key result:** the more subjective the task or the example, the worse a model trained on synthetic data performed.
- **How the paper uses it:** the precedent for Section 5.3, where heads trained on Claude-written tickets lose to Claude (Section 2).

## Two studies build narrow small models, the design closest to the heads

### 18. Lee et al. (2021): micromodels, many narrow models combined into one explainable system

- **Exact title:** Micromodels for Efficient, Explainable, and Reusable Systems: A Case Study on Mental Health
- **Published in:** Findings of EMNLP 2021; arXiv:2109.13770.
- **What it did:** built systems from small models that each detect one behavior, tested on depression, PTSD and suicide-risk tasks.
- **Key result:** the systems scored strongly even with little data, and they were more interpretable than the alternatives.
- **How the paper uses it:** the closest design precedent, many narrow models in place of one general model (Section 2).

### 19. Buckmann and Hill (2024): logistic regression on small-LLM embeddings matches GPT-4 with tens of labels

- **Exact title:** Logistic Regression makes small LLMs strong and explainable "tens-of-shot" classifiers
- **Published in:** arXiv:2408.03414.
- **What it did:** fit penalized logistic regression on a small LLM's embeddings across 17 sentence classification tasks.
- **Key result:** with tens of labeled examples it equaled a large LLM and usually beat it.
- **How the paper uses it:** the nearest method to the heads, which drop the neural network entirely (Section 2).
- **The paper's version:** Section 2 says the method is on par with GPT-4 "and sometimes better", where the abstract says "usually".

## Four statistics papers supply the intervals and tests

### 20. Berg-Kirkpatrick et al. (2012): the paired bootstrap for NLP comparisons

- **Exact title:** An Empirical Investigation of Statistical Significance in NLP
- **Published in:** EMNLP-CoNLL 2012, pages 995–1005.
- **What it did:** studied paired significance tests across NLP tasks, using many past systems' outputs and variants of single systems.
- **Key result:** it relates significance to the size of a gain, the test set and the systems' similarity, and tests whether it holds across domains.
- **How the paper uses it:** the paired bootstrap behind every 95% interval (Section 3.4).

### 21. Dror et al. (2018): a protocol for choosing a significance test in NLP

- **Exact title:** The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing
- **Published in:** ACL 2018, pages 1383–1392.
- **What it did:** set out how NLP tasks, setups and metrics should shape the choice of significance test.
- **Key result:** a simple protocol for picking a test, and a survey showing that 2017 ACL and TACL papers often ignored or misused significance testing.
- **How the paper uses it:** the paired approximate-randomization test behind each p-value (Section 3.4).

### 22. Holm (1979): the correction behind the 6 significant questions

- **Exact title:** A Simple Sequentially Rejective Multiple Test Procedure
- **Published in:** *Scandinavian Journal of Statistics* 6(2):65–70.
- **What it did:** introduced a procedure that rejects hypotheses one at a time until no further rejection is possible.
- **Key result:** it holds the chance of any false rejection at the chosen level for any mix of true hypotheses, with more power than Bonferroni.
- **How the paper uses it:** corrects the 15 per-question p-values (Sections 3.4, 4.1, 4.3 and 5.5).

### 23. McNemar (1947): the test on chats where only one system is right

- **Exact title:** Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages
- **Published in:** *Psychometrika* 12(2):153–157.
- **What it did:** presented two formulas for judging whether two correlated proportions differ.
- **Key result:** one formula has a chi-square equivalent, the test now known by McNemar's name.
- **How the paper uses it:** the exact test on chats where only one system is right (Sections 3.4, 4.1, 4.2 and 4.3).

## One software paper names the library behind the heads

### 24. Pedregosa et al. (2011): scikit-learn, the library the heads run on

- **Exact title:** Scikit-learn: Machine Learning in Python
- **Published in:** *Journal of Machine Learning Research* 12(85):2825–2830; arXiv:1201.0490.
- **What it did:** packaged a wide range of machine-learning algorithms for medium-scale problems in one Python module.
- **Key result:** a library built for ease of use, performance and a consistent API, under a BSD license that allows commercial use.
- **How the paper uses it:** version 1.7.2 supplies the TF-IDF features and the logistic regressions (Section 3.3).


**13 of the 18 related studies favor a trained model, and the one that favors a prompted LLM gave it retrieved examples, which this paper did not.**


