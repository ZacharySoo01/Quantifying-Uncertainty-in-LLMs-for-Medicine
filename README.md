# Uncertainty-Quantification-in-LLMs-for-Medical-Domain
Capstone Project by Zachary Soo
## Overview
In the medical domain, an important topic in the application of Large Language Models (LLMs) is **uncertainty**, as the issue of hallucination influences LLM certainty in its produced responses to a prompt.
This project explores two uncertainty quantification methods, Shifting Attention to Relevance (SAR) and Semantic Entropy, and applies and compares these methods to various LLMs on the PubMedQA dataset.

## Goal
How do different methods compare in quantifying how certain an LLM is in providing answers to medical questions (PubMedQA)?

* In answering this question, the LLMs that were used were OPT-2.7b, GPT-NEO-2.7b, Llama-3.2-1b, Phi-2-2.7b, and gemma-2b
* The dataset that was used was the PubMedQA dataset
* Experiments were run on a machine with 4 NVIDIA RTX 5000's and an Intel(R) Core(TM) i9-9960X CPU @ 3.10GHz

## Prerequisites 
1) Install miniconda at https://docs.anaconda.com/miniconda/
2) Create a HuggingFace account and obtain access to Phi-2, Llama-3.2, and Gemma
3) Login to the HuggingFace account via huggingface-cli
```shell
huggingface-cli login
```
4) Create a wandb account

## Running SAR experiments
1) In the SAR directory, Create and activate the SAR environment
```shell
conda create -n SAR python=3.10.15
conda activate SAR
```
2) Install requirements
```shell
pip install -r requirements.txt
```
2) Go to src directory
```shell
cd src
```
3) Run scripts
```shell
sh scripts/pubmedqa/ue_pipeline-opt-2.7b
sh scripts/pubmedqa/ue_pipeline-gpt-neo-2.7b
sh scripts/pubmedqa/ue_pipeline-Llama-3.2-1b
sh scripts/pubmedqa/ue_pipeline-phi2-2.7b
sh scripts/pubmedqa/ue_pipeline-gemma-2b
```
4) Results will be tracked in the experiments folder
```shell
cd ..
cd experiments
```
## Running Semantic Entropy experiments
1) In the semantic_uncertainty directory, run the following commands to create and activate the semantic_uncertainty environment
```shell
conda env create -f environment.yaml -n semantic_uncertainty 
conda activate semantic_uncertainty
pip install huggingface --upgrade
```
2) Run scripts
```shell
python semantic_uncertainty/generate_answers.py --model_name=opt-2.7b --dataset=pubmedqa
python semantic_uncertainty/generate_answers.py --model_name=gpt-neo-2.7b --dataset=pubmedqa
python semantic_uncertainty/generate_answers.py --model_name=Llama-3.2-1B --dataset=pubmedqa
python semantic_uncertainty/generate_answers.py --model_name=phi-2 --dataset=pubmedqa
python semantic_uncertainty/generate_answers.py --model_name=gemma-2b --dataset=pubmedqa
```
When script is running, it will prompt for a wandb api key. Choose option 2 and enter in your api key, and the script will continue to run.

3) Results will be tracked in wandb

## References
```shell
@inproceedings{duan2024shifting,
  title={Shifting Attention to Relevance: Towards the Predictive Uncertainty Quantification of Free-Form Large Language Models},
  author={Duan, Jinhao and Cheng, Hao and Wang, Shiqi and Zavalny, Alex and Wang, Chenan and Xu, Renjing and Kailkhura, Bhavya and Xu, Kaidi},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={5050--5063},
  year={2024}
}
```
```shell
@inproceedings{
  title={Detecting Hallucinations in Large Language Models Using Semantic Entropy},
  author={Farquhar, Sebastian and Kossen, Jannik and Kuhn, Lorenz and Gal, Yarin},
  booktitle={Nature},
  pages={625--630},
  year={2024}
}
```
Source code repositories:
* https://github.com/jlko/semantic_uncertainty
* https://github.com/jinhaoduan/SAR
