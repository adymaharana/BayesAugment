## Adversarial Augmentation Policy Search for Domain and Cross-Lingual Generalization in Reading Comprehension

PyTorch code for the Findings of EMNLP 2020 paper "Adversarial Augmentation Policy Search for Domain and Cross-Lingual Generalization in Reading Comprehension".

#### Training:
In order to run BayesAugment for SQuaD v2.0 using Roberta-Base, please run the following:\
```python search_augmentation_policy_bayesopt.py```

This code was run on a system with 4 2080 Ti GPUs.

#### Additional Information:
The BayesAugment codebase is heavily derived from the [original Adversarial RC codebase](https://github.com/robinjia/adversarial-squad). \
Additionally, to prepare the adversarial dataset, the pipeline uses/requires the following repositories:\
[nectar](https://github.com/robinjia/nectar) (used in original code)\
[pattern](https://github.com/clips/pattern) (used in original code)\
[Syntactic Paraphrasing Networks](https://github.com/miyyer/scpn) (To generate syntactic paraphrases)\
[Semantic Paraphrases](https://github.com/nesl/nlp_adversarial_examples) (To generate semantic paraphrases)

