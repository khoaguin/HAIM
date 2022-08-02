# Holistic Artificial Intelligence in Medicine

This repository contains the code to replicate the data processing, modeling and reporting of our Holistic AI in Medicine (HAIM) accepted for publication in Nature npj Digital Medicine (Soenksen LR, Ma Y, Zeng C et al. 2022). You can find a pre-print in https://arxiv.org/abs/2202.12998. 

## Integrated multimodal artificial intelligence framework for healthcare applications
Luis R. Soenksen, Yu Ma, Cynthia Zeng, Léonard Boussioux, Kimberly Villalobos Carballo, Liangyuan Na, Holly M. Wiberg, Michael L. Li, Ignacio Fuentes, Dimitris Bertsimas

Artificial intelligence (AI) systems hold great promise to improve healthcare over the next decades. Specifically, AI systems leveraging multiple data sources and input modalities are poised to become a viable method to deliver more accurate results and deployable pipelines across a wide range of applications. In this work, we propose and evaluate a unified Holistic AI in Medicine (HAIM) framework to facilitate the generation and testing of AI systems that leverage multimodal inputs. Our approach uses generalizable data pre-processing and machine learning modeling stages that can be readily adapted for research and deployment in healthcare environments. We evaluate our HAIM framework by training and characterizing 14,324 independent models based on HAIM-MIMIC-MM, a multimodal clinical database (N=34,537 samples) containing 7,279 unique hospitalizations and 6,485 patients, spanning all possible input combinations of 4 data modalities (i.e., tabular, time-series, text, and images), 11 unique data sources and 12 predictive tasks. We show that this framework can consistently and robustly produce models that outperform similar single-source approaches across various healthcare demonstrations (by 6-33%), including 10 distinct chest pathology diagnoses, along with length-of-stay and 48-hour mortality predictions. We also quantify the contribution of each modality and data source using Shapley values, which demonstrates the heterogeneity in data modality importance and the necessity of multimodal inputs across different healthcare-relevant tasks. The generalizable properties and flexibility of our Holistic AI in Medicine (HAIM) framework could offer a promising pathway for future multimodal predictive systems in clinical and operational healthcare settings.

## Code

The code uses Python3.6.9 and is separated into four sections:

0 - Software Package requirement

1 - Data Preprocessing

2 - Modeling of our three tasks: mortality prediction, length of stay prediction, chest pathology classification

3 - Result Generating: Including reporting of the AUROC, AUPRC, F1 scores, as well as code to generate the plots reported in the paper.


Please be advised that sufficient RAM or cluster access to parallel processing is needed to run these experiments. 
