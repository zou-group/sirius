<!--- BADGES: START --->

[![GitHub license](https://img.shields.io/badge/License-MIT-blue.svg)][#license-gh-package]
[![Arxiv](https://img.shields.io/badge/arXiv-2406.07496-B31B1B.svg)][#arxiv-paper-package]


[#license-gh-package]: https://lbesson.mit-license.org/
[#arxiv-paper-package]: https://arxiv.org/pdf/2502.04780

<!--- BADGES: END --->
 
## SiriuS: Self-improving Multi-agent Systems via Bootstrapped Reasoning (NeurIPS 2025)
 
This is the repository for the paper [**SiriuS: Self-improving Multi-agent Systems via Bootstrapped Reasoning**](https://arxiv.org/pdf/2502.04780) (NeurIPS 2025).

SIRIUS is a self-improving multi-agent framework that continuously enhances reasoning ability by maintaining an experience library of successful trajectories and bootstrapping from failed ones.

We support three main multi-agent settings, each with its own directory:

- `Problem_solving/` – collaborative QA (College Physics/Chemistry, PubMedQA-style)  
- `Actor_Critic/` – Actor + Judgment + Critic for iterative refinement  
- `Competitive/` – negotiation / game-theoretic interactions  


![Analogy with Torch](assets/task.png)



### Setup

#### Clone the repo

```bash
git clone https://github.com/zou-group/sirius.git
cd sirius
```

#### Create environment & install dependencies

```bash
conda create -n sirius python=3.10
conda activate sirius
conda env create -f environment.yml
```

#### Configure API access
Set your keys as environment variables or in a config file as used by the codebase, for example:
```bash
export OPENAI_API_KEY=...
```

### Repository Overview


* `Problem_solving/`
  Pipelines for college-level reasoning & biomedical QA:
  * College Physics / College Chemistry
  * PubMedQA-style question answering (long context + question)

* `Actor_Critic/`
  Pipelines for the Actor–Judgment–Critic setting:
  * Actor proposes an answer
  * Judgment agent decides correct / incorrect
  * Critic writes feedback and guides regeneration

* `Competitive/`
  Pipelines for competitive games:
  * Resource Exchange
  * Sell & Buy
  * Ultimatum
Each is a two-player turn-based game with utilities defined in the paper.

#### Data Format & Trajectories

SiriuS operates on trajectories:

* A trajectory is the full interaction between agents for one task instance:

* Input question / context

* Intermediate messages from each agent (Physicist, Mathematician, Summarizer, Actor, Critic, etc.)

* Final answer(s) or game outcome

* Reward signal(s) (accuracy or utility)

### Quick Start

#### Collect Raw Multi-Agent Trajectories

First, run the multi-agent system (with base models) on your tasks and log the full interaction.

A sample training dataset (for physics problem solving) is already provided at:
```bash
dataset/phy_train.jsonl
```
Each line of this file is one training example (e.g., one physics problem) that the multi-agent system will solve.

Put your training and eval data at 
```bash
dataset/{subject}_train.jsonl
dataset/{subject}_test.jsonl
```

Each subdirectory provides task-specific drivers to:

* Load the dataset 
* Instantiate the appropriate agent graph (see the paper for structures)

  ```bash
  Problem_solving/PhyChem/agent.py  
  ```

* solve the problems, collect full trajectories
  ```bash
  python Problem_solving/PhyChem/get_a_sol.py --model='gpt-3.5-turbo' --task='MMLU_physics'  --prompt_type='multi_agent' --mode='generate' --subject='phy'
  ```
#### Filter Trajectories
```bash 
python libs/merge.py
```
#### Augment Failed Trajectories

First, generate feedback for trajectories where the agents produced incorrect solutions:
  ```bash
  python Problem_solving/PhyChem/get_b_feedback.py --model='gpt-3.5-turbo' --task='MMLU_physics'  --prompt_type='multi_agent' --mode='generate' --subject='phy'
  ```

Then, regenerate improved trajectories conditioned on this feedback:
  ```bash
  python Problem_solving/PhyChem/get_c_regenerate.py --model='gpt-3.5-turbo' --task='MMLU_physics'  --prompt_type='multi_agent' --mode='generate' --subject='phy'
  ```


#### Fine-Tune Agents on the Library

We use the OpenAI Supervised Fine-Tuning (SFT) API in our example, but you can plug in any fine-tuning framework of your choice using the constructed experience library:


```bash
python Problem_solving/PhyChem/get_finetune_data.py
python Problem_solving/PhyChem/fine_tune.py
```



```bibtex
@article{zhao2025sirius,
  title={SiriuS: Self-improving Multi-agent Systems via Bootstrapped Reasoning},
  author={Zhao, Wanjia and Yuksekgonul, Mert and Wu, Shirley and Zou, James},
  journal={arXiv preprint arXiv:2502.04780},
  year={2025}
}
```