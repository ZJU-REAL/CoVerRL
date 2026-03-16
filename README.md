<div align="center">
     <h1>CoVerRL: Breaking the Consensus Trap in Label-Free Reasoning via Reinforcement Learning</h1>
</div>

<div align='center'> 

[Teng Pan](mailto:pt6@zju.edu.cn)<sup>1,2,\*</sup>, &nbsp;
[Yuchen Yan]<sup>1</sup>, &nbsp;
Zixuan Wang<sup>1,2</sup>, &nbsp;
Ruiqing Zhang<sup>2</sup>, &nbsp;
<br>
Guiyang Hou<sup>1</sup>, &nbsp;
Wenqi Zhang<sup>1</sup>, &nbsp;
Weiming Lu<sup>1</sup>, &nbsp;
Jun Xiao<sup>1</sup>, &nbsp;
[Yongliang Shen](mailto:syl@zju.edu.cn)<sup>1,†</sup>  

<sup>1</sup>Zhejiang University, &nbsp;
<sup>2</sup>Baidu Inc.\
<em>Preprint. Under review.</em>  
<sup>*</sup>Contribution during internship at Baidu Inc. <sup>†</sup>Corresponding Author
</div>


<!-- <p align="center">
<img src="docs/static/images/arxiv_logo.png" alt="arXiv" height="14"> <a href="https://arxiv.org/abs/2602.06960">Arxiv</a> 
| 📑 <a href="https://zju-real.github.io/InftyThink-Plus/">WebPage</a> 
<br>
</p> -->

<!-- ## News 🔥🔥
- **2026.03.16:** We release our paper. -->

## Overview 🦾🦾
<img src="docs/static/images/method.png" width="100%"/>

Label-free reinforcement learning for LLMs typically adopts majority voting to generate pseudo-labels, but suffers from a consensus trap—output diversity collapses during training, leading the model to confidently reinforce systematic self-consistent errors. To address this issue, we propose CoVerRL, a novel framework that unifies generator and verifier roles into a single model via multi-turn reinforcement learning, enabling their mutual bootstrapping and co-evolution without external ground-truth labels.

Our contributions can be summarized as follows:

- We identify the consensus trap in majority voting based label-free RL, where diversity collapse causes
reward accuracy degradation as models become overconfident in systematic errors, explaining why such
methods eventually stagnate.

- We propose CoVerRL, a co-evolution framework that unifies generation and verification into a multi-turn
RL process, enabling mutual bootstrapping where each capability supervises improvement of the other
without external labels.

- We validate CoVerRL across Qwen and Llama model families, demonstrating 4-6% improvements over
label-free baselines on mathematical reasoning benchmarks while producing verifiers that generalize well to
held-out evaluation.

## Training Dynamic 📈📈
<p align="center">
  <img src="docs/static/images/intro.png" width="60%" alt="Introduction Overview"/>
</p>

Training dynamics of reward/label accuracy for TTRL and CoVerRL on Qwen3-1.7B-Base. CoVerRL maintains reward accuracy above around 90% and boosts label accuracy via generator-verifier co-evolution, while TTRL faces reward accuracy degradation and stagnant label accuracy due to the consensus trap.

<!-- ## Main Results 📊📊
<p align="center">
  <img src="docs/static/images/main_results.png" width="100%" alt="Introduction Overview"/>
</p> -->


## QuickStart 🎯🎯
### Installation
This repository is based on verl v0.6.x branch. Please refer to 
<a href='https://verl.readthedocs.io/en/latest/start/install.html'>verl installation</a> for setup instructions. Additionally, install <a href='https://github.com/huggingface/Math-Verify'>Math-Verify</a> as the verifier: ``` pip install math-verify ```


### TTRL baseline
```bash
cd verl
bash recipe/cover_rl/scripts/gpu/ttrl_baseline.sh
```
### CoVerRL
```bash
cd verl
bash recipe/cover_rl/scripts/gpu/cover_rl.sh
```

<!-- ## Citation

If you find our work helpful, feel free to give us a cite.

```
@misc{yan2026inftythinkplus,
      title={CoVerRL: Breaking the Consensus Trap in Label-Free Reasoning via Generator-Verifier Co-Evolution}, 
      author={Yuchen Yan and Liang Jiang and Jin Jiang and Shuaicheng Li and Zujie Wen and Zhiqiang Zhang and Jun Zhou and Jian Shao and Yueting Zhuang and Yongliang Shen},
      year={2026},
      eprint={2602.06960},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.06960}, 
}
``` -->

## Contact Us
If you have any questions, please contact us by email: 
pt6@zju.edu.cn
