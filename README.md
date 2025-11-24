# DeepSeek-R1
<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/logo.svg?raw=true" width="60%" alt="DeepSeek-V3" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://www.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/badge.svg?raw=true" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://chat.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/🤖%20Chat-DeepSeek%20R1-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/deepseek-ai" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://discord.gg/Tc7c45Zzu5" target="_blank" style="margin: 2px;">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-DeepSeek%20AI-7289da?logo=discord&logoColor=white&color=7289da" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/qr.jpeg?raw=true" target="_blank" style="margin: 2px;">
    <img alt="Wechat" src="https://img.shields.io/badge/WeChat-DeepSeek%20AI-brightgreen?logo=wechat&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://twitter.com/deepseek_ai" target="_blank" style="margin: 2px;">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://github.com/deepseek-ai/DeepSeek-R1/blob/main/LICENSE-CODE" style="margin: 2px;">
    <img alt="Code License" src="https://img.shields.io/badge/Code_License-MIT-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/deepseek-ai/DeepSeek-R1/blob/main/LICENSE-MODEL" style="margin: 2px;">
    <img alt="Model License" src="https://img.shields.io/badge/Model_License-Model_Agreement-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>


<p align="center">
  <a href="https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf"><b>Paper Link</b>👁️</a>
</p>


## 1. Introduction

We introduce our first-generation reasoning models, DeepSeek-R1-Zero and DeepSeek-R1. 
DeepSeek-R1-Zero, a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step, demonstrated remarkable performance on reasoning.
With RL, DeepSeek-R1-Zero naturally emerged with numerous powerful and interesting reasoning behaviors.
However, DeepSeek-R1-Zero encounters challenges such as endless repetition, poor readability, and language mixing. To address these issues and further enhance reasoning performance,
we introduce DeepSeek-R1, which incorporates cold-start data before RL.
DeepSeek-R1 achieves performance comparable to OpenAI-o1 across math, code, and reasoning tasks. 
To support the research community, we have open-sourced DeepSeek-R1-Zero, DeepSeek-R1, and six dense models distilled from DeepSeek-R1 based on Llama and Qwen. DeepSeek-R1-Distill-Qwen-32B outperforms OpenAI-o1-mini across various benchmarks, achieving new state-of-the-art results for dense models.

**NOTE: Before running DeepSeek-R1 series models locally, we kindly recommend reviewing the [Usage Recommendation](#usage-recommendations) section.**

<p align="center">
  <img width="80%" src="figures/benchmark.jpg">
</p>

## 2. Model Summary

---

**Post-Training: Large-Scale Reinforcement Learning on the Base Model**

-  We directly apply reinforcement learning (RL) to the base model without relying on supervised fine-tuning (SFT) as a preliminary step. This approach allows the model to explore chain-of-thought (CoT) for solving complex problems, resulting in the development of DeepSeek-R1-Zero. DeepSeek-R1-Zero demonstrates capabilities such as self-verification, reflection, and generating long CoTs, marking a significant milestone for the research community. Notably, it is the first open research to validate that reasoning capabilities of LLMs can be incentivized purely through RL, without the need for SFT. This breakthrough paves the way for future advancements in this area.

-   We introduce our pipeline to develop DeepSeek-R1. The pipeline incorporates two RL stages aimed at discovering improved reasoning patterns and aligning with human preferences, as well as two SFT stages that serve as the seed for the model's reasoning and non-reasoning capabilities.
    We believe the pipeline will benefit the industry by creating better models. 

---

**Distillation: Smaller Models Can Be Powerful Too**

-  We demonstrate that the reasoning patterns of larger models can be distilled into smaller models, resulting in better performance compared to the reasoning patterns discovered through RL on small models. The open source DeepSeek-R1, as well as its API, will benefit the research community to distill better smaller models in the future. 
- Using the reasoning data generated by DeepSeek-R1, we fine-tuned several dense models that are widely used in the research community. The evaluation results demonstrate that the distilled smaller dense models perform exceptionally well on benchmarks. We open-source distilled 1.5B, 7B, 8B, 14B, 32B, and 70B checkpoints based on Qwen2.5 and Llama3 series to the community.

## 3. Model Downloads

### DeepSeek-R1 Models

<div align="center">

| **Model** | **#Total Params** | **#Activated Params** | **Context Length** | **Download** |
| :------------: | :------------: | :------------: | :------------: | :------------: |
| DeepSeek-R1-Zero | 671B | 37B | 128K   | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero)   |
| DeepSeek-R1   | 671B | 37B |  128K   | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1)   |

</div>

DeepSeek-R1-Zero & DeepSeek-R1 are trained based on DeepSeek-V3-Base. 
For more details regarding the model architecture, please refer to [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) repository.

### DeepSeek-R1-Distill Models

<div align="center">

| **Model** | **Base Model** | **Download** |
| :------------: | :------------: | :------------: |
| DeepSeek-R1-Distill-Qwen-1.5B  | [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)   |
| DeepSeek-R1-Distill-Qwen-7B  | [Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B) | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)   |
| DeepSeek-R1-Distill-Llama-8B  | [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)   |
| DeepSeek-R1-Distill-Qwen-14B   | [Qwen2.5-14B](https://huggingface.co/Qwen/Qwen2.5-14B) | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)   |
|DeepSeek-R1-Distill-Qwen-32B  | [Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B) | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)   |
| DeepSeek-R1-Distill-Llama-70B  | [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)   |

</div>

DeepSeek-R1-Distill models are fine-tuned based on open-source models, using samples generated by DeepSeek-R1.
We slightly change their configs and tokenizers. Please use our setting to run these models.

## 4. Evaluation Results

### DeepSeek-R1-Evaluation
 For all our models, the maximum generation length is set to 32,768 tokens. For benchmarks requiring sampling, we use a temperature of $0.6$, a top-p value of $0.95$, and generate 64 responses per query to estimate pass@1.
<div align="center">


| Category | Benchmark (Metric) | Claude-3.5-Sonnet-1022 | GPT-4o 0513 | DeepSeek V3 | OpenAI o1-mini | OpenAI o1-1217 | DeepSeek R1 |
|----------|-------------------|----------------------|------------|--------------|----------------|------------|--------------|
| | Architecture | - | - | MoE | - | - | MoE |
| | # Activated Params | - | - | 37B | - | - | 37B |
| | # Total Params | - | - | 671B | - | - | 671B |
| English | MMLU (Pass@1) | 88.3 | 87.2 | 88.5 | 85.2 | **91.8** | 90.8 |
| | MMLU-Redux (EM) | 88.9 | 88.0 | 89.1 | 86.7 | - | **92.9** |
| | MMLU-Pro (EM) | 78.0 | 72.6 | 75.9 | 80.3 | - | **84.0** |
| | DROP (3-shot F1) | 88.3 | 83.7 | 91.6 | 83.9 | 90.2 | **92.2** |
| | IF-Eval (Prompt Strict) | **86.5** | 84.3 | 86.1 | 84.8 | - | 83.3 |
| | GPQA-Diamond (Pass@1) | 65.0 | 49.9 | 59.1 | 60.0 | **75.7** | 71.5 |
| | SimpleQA (Correct) | 28.4 | 38.2 | 24.9 | 7.0 | **47.0** | 30.1 |
| | FRAMES (Acc.) | 72.5 | 80.5 | 73.3 | 76.9 | - | **82.5** |
| | AlpacaEval2.0 (LC-winrate) | 52.0 | 51.1 | 70.0 | 57.8 | - | **87.6** |
| | ArenaHard (GPT-4-1106) | 85.2 | 80.4 | 85.5 | 92.0 | - | **92.3** |
| Code | LiveCodeBench (Pass@1-COT) | 33.8 | 34.2 | - | 53.8 | 63.4 | **65.9** |
| | Codeforces (Percentile) | 20.3 | 23.6 | 58.7 | 93.4 | **96.6** | 96.3 |
| | Codeforces (Rating) | 717 | 759 | 1134 | 1820 | **2061** | 2029 |
| | SWE Verified (Resolved) | **50.8** | 38.8 | 42.0 | 41.6 | 48.9 | 49.2 |
| | Aider-Polyglot (Acc.) | 45.3 | 16.0 | 49.6 | 32.9 | **61.7** | 53.3 |
| Math | AIME 2024 (Pass@1) | 16.0 | 9.3 | 39.2 | 63.6 | 79.2 | **79.8** |
| | MATH-500 (Pass@1) | 78.3 | 74.6 | 90.2 | 90.0 | 96.4 | **97.3** |
| | CNMO 2024 (Pass@1) | 13.1 | 10.8 | 43.2 | 67.6 | - | **78.8** |
| Chinese | CLUEWSC (EM) | 85.4 | 87.9 | 90.9 | 89.9 | - | **92.8** |
| | C-Eval (EM) | 76.7 | 76.0 | 86.5 | 68.9 | - | **91.8** |
| | C-SimpleQA (Correct) | 55.4 | 58.7 | **68.0** | 40.3 | - | 63.7 |

</div>


### Distilled Model Evaluation


<div align="center">

| Model                                    | AIME 2024 pass@1 | AIME 2024 cons@64 | MATH-500 pass@1 | GPQA Diamond pass@1 | LiveCodeBench pass@1 | CodeForces rating |
|------------------------------------------|------------------|-------------------|-----------------|----------------------|----------------------|-------------------|
| GPT-4o-0513                          | 9.3              | 13.4              | 74.6            | 49.9                 | 32.9                 | 759               |
| Claude-3.5-Sonnet-1022             | 16.0             | 26.7                 | 78.3            | 65.0                 | 38.9                 | 717               |
| o1-mini                              | 63.6             | 80.0              | 90.0            | 60.0                 | 53.8                 | **1820**          |
| QwQ-32B-Preview                              | 44.0             | 60.0                 | 90.6            | 54.5               | 41.9                 | 1316              |
| DeepSeek-R1-Distill-Qwen-1.5B       | 28.9             | 52.7              | 83.9            | 33.8                 | 16.9                 | 954               |
| DeepSeek-R1-Distill-Qwen-7B          | 55.5             | 83.3              | 92.8            | 49.1                 | 37.6                 | 1189              |
| DeepSeek-R1-Distill-Qwen-14B         | 69.7             | 80.0              | 93.9            | 59.1                 | 53.1                 | 1481              |
| DeepSeek-R1-Distill-Qwen-32B        | **72.6**         | 83.3              | 94.3            | 62.1                 | 57.2                 | 1691              |
| DeepSeek-R1-Distill-Llama-8B         | 50.4             | 80.0              | 89.1            | 49.0                 | 39.6                 | 1205              |
| DeepSeek-R1-Distill-Llama-70B        | 70.0             | **86.7**          | **94.5**        | **65.2**             | **57.5**             | 1633              |

</div>


## 5. Chat Website & API Platform
You can chat with DeepSeek-R1 on DeepSeek's official website: [chat.deepseek.com](https://chat.deepseek.com), and switch on the button "DeepThink"

We also provide OpenAI-Compatible API at DeepSeek Platform: [platform.deepseek.com](https://platform.deepseek.com/)

## 6. How to Run Locally

### DeepSeek-R1 Models

Please visit [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) repo for more information about running DeepSeek-R1 locally.

### DeepSeek-R1-Distill Models

DeepSeek-R1-Distill models can be utilized in the same manner as Qwen or Llama models.

For instance, you can easily start a service using [vLLM](https://github.com/vllm-project/vllm):

```shell
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager
```

You can also easily start a service using [SGLang](https://github.com/sgl-project/sglang)

```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --trust-remote-code --tp 2
```

### Usage Recommendations

**We recommend adhering to the following configurations when utilizing the DeepSeek-R1 series models, including benchmarking, to achieve the expected performance:**

1. Set the temperature within the range of 0.5-0.7 (0.6 is recommended) to prevent endless repetitions or incoherent outputs.
2. **Avoid adding a system prompt; all instructions should be contained within the user prompt.**
3. For mathematical problems, it is advisable to include a directive in your prompt such as: "Please reason step by step, and put your final answer within \boxed{}."
4. When evaluating model performance, it is recommended to conduct multiple tests and average the results.

## 7. License
This code repository and the model weights are licensed under the [MIT License](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/LICENSE).
DeepSeek-R1 series support commercial use, allow for any modifications and derivative works, including, but not limited to, distillation for training other LLMs. Please note that:
- DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Qwen-14B and DeepSeek-R1-Distill-Qwen-32B are derived from [Qwen-2.5 series](https://github.com/QwenLM/Qwen2.5), which are originally licensed under [Apache 2.0 License](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/LICENSE), and now finetuned with 800k samples curated with DeepSeek-R1.
- DeepSeek-R1-Distill-Llama-8B is derived from Llama3.1-8B-Base and is originally licensed under [llama3.1 license](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/LICENSE).
- DeepSeek-R1-Distill-Llama-70B is derived from Llama3.3-70B-Instruct and is originally licensed under [llama3.3 license](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/LICENSE).

## 8. Citation
```
@misc{deepseekai2025deepseekr1incentivizingreasoningcapability,
      title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning}, 
      author={DeepSeek-AI and Daya Guo and Dejian Yang and Haowei Zhang and Junxiao Song and Ruoyu Zhang and Runxin Xu and Qihao Zhu and Shirong Ma and Peiyi Wang and Xiao Bi and Xiaokang Zhang and Xingkai Yu and Yu Wu and Z. F. Wu and Zhibin Gou and Zhihong Shao and Zhuoshu Li and Ziyi Gao and Aixin Liu and Bing Xue and Bingxuan Wang and Bochao Wu and Bei Feng and Chengda Lu and Chenggang Zhao and Chengqi Deng and Chenyu Zhang and Chong Ruan and Damai Dai and Deli Chen and Dongjie Ji and Erhang Li and Fangyun Lin and Fucong Dai and Fuli Luo and Guangbo Hao and Guanting Chen and Guowei Li and H. Zhang and Han Bao and Hanwei Xu and Haocheng Wang and Honghui Ding and Huajian Xin and Huazuo Gao and Hui Qu and Hui Li and Jianzhong Guo and Jiashi Li and Jiawei Wang and Jingchang Chen and Jingyang Yuan and Junjie Qiu and Junlong Li and J. L. Cai and Jiaqi Ni and Jian Liang and Jin Chen and Kai Dong and Kai Hu and Kaige Gao and Kang Guan and Kexin Huang and Kuai Yu and Lean Wang and Lecong Zhang and Liang Zhao and Litong Wang and Liyue Zhang and Lei Xu and Leyi Xia and Mingchuan Zhang and Minghua Zhang and Minghui Tang and Meng Li and Miaojun Wang and Mingming Li and Ning Tian and Panpan Huang and Peng Zhang and Qiancheng Wang and Qinyu Chen and Qiushi Du and Ruiqi Ge and Ruisong Zhang and Ruizhe Pan and Runji Wang and R. J. Chen and R. L. Jin and Ruyi Chen and Shanghao Lu and Shangyan Zhou and Shanhuang Chen and Shengfeng Ye and Shiyu Wang and Shuiping Yu and Shunfeng Zhou and Shuting Pan and S. S. Li and Shuang Zhou and Shaoqing Wu and Shengfeng Ye and Tao Yun and Tian Pei and Tianyu Sun and T. Wang and Wangding Zeng and Wanjia Zhao and Wen Liu and Wenfeng Liang and Wenjun Gao and Wenqin Yu and Wentao Zhang and W. L. Xiao and Wei An and Xiaodong Liu and Xiaohan Wang and Xiaokang Chen and Xiaotao Nie and Xin Cheng and Xin Liu and Xin Xie and Xingchao Liu and Xinyu Yang and Xinyuan Li and Xuecheng Su and Xuheng Lin and X. Q. Li and Xiangyue Jin and Xiaojin Shen and Xiaosha Chen and Xiaowen Sun and Xiaoxiang Wang and Xinnan Song and Xinyi Zhou and Xianzu Wang and Xinxia Shan and Y. K. Li and Y. Q. Wang and Y. X. Wei and Yang Zhang and Yanhong Xu and Yao Li and Yao Zhao and Yaofeng Sun and Yaohui Wang and Yi Yu and Yichao Zhang and Yifan Shi and Yiliang Xiong and Ying He and Yishi Piao and Yisong Wang and Yixuan Tan and Yiyang Ma and Yiyuan Liu and Yongqiang Guo and Yuan Ou and Yuduan Wang and Yue Gong and Yuheng Zou and Yujia He and Yunfan Xiong and Yuxiang Luo and Yuxiang You and Yuxuan Liu and Yuyang Zhou and Y. X. Zhu and Yanhong Xu and Yanping Huang and Yaohui Li and Yi Zheng and Yuchen Zhu and Yunxian Ma and Ying Tang and Yukun Zha and Yuting Yan and Z. Z. Ren and Zehui Ren and Zhangli Sha and Zhe Fu and Zhean Xu and Zhenda Xie and Zhengyan Zhang and Zhewen Hao and Zhicheng Ma and Zhigang Yan and Zhiyu Wu and Zihui Gu and Zijia Zhu and Zijun Liu and Zilin Li and Ziwei Xie and Ziyang Song and Zizheng Pan and Zhen Huang and Zhipeng Xu and Zhongyu Zhang and Zhen Zhang},
      year={2025},
      eprint={2501.12948},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.12948}, 
}

```

## 9. Contact
If you have any questions, please raise an issue or contact us at [service@deepseek.com](service@deepseek.com).


### Automated Update - Sat Feb  1 06:21:26 UTC 2025 🚀


### Automated Update - Sat Feb  1 06:25:47 UTC 2025 🚀


### Automated Update - Sat Feb  1 06:37:20 UTC 2025 🚀


### Automated Update - Sat Feb  1 06:42:33 UTC 2025 🚀


### Automated Update - Sat Feb  1 06:54:04 UTC 2025 🚀


### Automated Update - Sat Feb  1 12:12:56 UTC 2025 🚀


### Automated Update - Sun Feb  2 00:41:07 UTC 2025 🚀


### Automated Update - Sun Feb  2 12:12:40 UTC 2025 🚀


### Automated Update - Mon Feb  3 00:40:02 UTC 2025 🚀


### Automated Update - Mon Feb  3 12:14:53 UTC 2025 🚀


### Automated Update - Tue Feb  4 00:38:53 UTC 2025 🚀


### Automated Update - Tue Feb  4 12:15:34 UTC 2025 🚀


### Automated Update - Wed Feb  5 00:39:07 UTC 2025 🚀


### Automated Update - Wed Feb  5 12:15:31 UTC 2025 🚀


### Automated Update - Thu Feb  6 00:39:26 UTC 2025 🚀


### Automated Update - Thu Feb  6 12:15:36 UTC 2025 🚀


### Automated Update - Fri Feb  7 00:39:17 UTC 2025 🚀


### Automated Update - Fri Feb  7 12:14:56 UTC 2025 🚀


### Automated Update - Sat Feb  8 00:38:18 UTC 2025 🚀


### Automated Update - Sat Feb  8 12:13:14 UTC 2025 🚀


### Automated Update - Sun Feb  9 00:42:02 UTC 2025 🚀


### Automated Update - Sun Feb  9 12:13:13 UTC 2025 🚀


### Automated Update - Mon Feb 10 00:40:34 UTC 2025 🚀


### Automated Update - Mon Feb 10 12:15:14 UTC 2025 🚀


### Automated Update - Tue Feb 11 00:39:23 UTC 2025 🚀


### Automated Update - Tue Feb 11 12:15:33 UTC 2025 🚀


### Automated Update - Wed Feb 12 00:39:19 UTC 2025 🚀


### Automated Update - Wed Feb 12 12:15:17 UTC 2025 🚀


### Automated Update - Thu Feb 13 00:39:42 UTC 2025 🚀


### Automated Update - Thu Feb 13 12:15:19 UTC 2025 🚀


### Automated Update - Fri Feb 14 00:39:16 UTC 2025 🚀


### Automated Update - Fri Feb 14 12:15:04 UTC 2025 🚀


### Automated Update - Sat Feb 15 00:38:43 UTC 2025 🚀


### Automated Update - Sat Feb 15 12:13:12 UTC 2025 🚀


### Automated Update - Sun Feb 16 00:43:12 UTC 2025 🚀


### Automated Update - Sun Feb 16 12:15:52 UTC 2025 🚀


### Automated Update - Mon Feb 17 00:41:53 UTC 2025 🚀


### Automated Update - Mon Feb 17 12:15:43 UTC 2025 🚀


### Automated Update - Tue Feb 18 00:39:01 UTC 2025 🚀


### Automated Update - Tue Feb 18 12:15:34 UTC 2025 🚀


### Automated Update - Wed Feb 19 00:39:28 UTC 2025 🚀


### Automated Update - Wed Feb 19 12:14:57 UTC 2025 🚀


### Automated Update - Thu Feb 20 00:39:59 UTC 2025 🚀


### Automated Update - Thu Feb 20 12:15:34 UTC 2025 🚀


### Automated Update - Fri Feb 21 00:39:57 UTC 2025 🚀


### Automated Update - Fri Feb 21 12:15:06 UTC 2025 🚀


### Automated Update - Sat Feb 22 00:38:23 UTC 2025 🚀


### Automated Update - Sat Feb 22 12:12:55 UTC 2025 🚀


### Automated Update - Sun Feb 23 00:42:59 UTC 2025 🚀


### Automated Update - Sun Feb 23 12:13:22 UTC 2025 🚀


### Automated Update - Mon Feb 24 00:41:37 UTC 2025 🚀


### Automated Update - Mon Feb 24 12:15:50 UTC 2025 🚀


### Automated Update - Tue Feb 25 00:40:30 UTC 2025 🚀


### Automated Update - Tue Feb 25 12:15:38 UTC 2025 🚀


### Automated Update - Wed Feb 26 00:40:14 UTC 2025 🚀


### Automated Update - Wed Feb 26 12:15:57 UTC 2025 🚀


### Automated Update - Thu Feb 27 00:40:28 UTC 2025 🚀


### Automated Update - Thu Feb 27 12:15:41 UTC 2025 🚀


### Automated Update - Fri Feb 28 00:40:39 UTC 2025 🚀


### Automated Update - Fri Feb 28 12:15:03 UTC 2025 🚀


### Automated Update - Sat Mar  1 00:44:00 UTC 2025 🚀


### Automated Update - Sat Mar  1 12:13:46 UTC 2025 🚀


### Automated Update - Sun Mar  2 00:43:46 UTC 2025 🚀


### Automated Update - Sun Mar  2 12:13:20 UTC 2025 🚀


### Automated Update - Mon Mar  3 00:42:18 UTC 2025 🚀


### Automated Update - Mon Mar  3 12:16:13 UTC 2025 🚀


### Automated Update - Tue Mar  4 00:41:09 UTC 2025 🚀


### Automated Update - Tue Mar  4 12:15:41 UTC 2025 🚀


### Automated Update - Wed Mar  5 00:41:16 UTC 2025 🚀


### Automated Update - Wed Mar  5 12:15:49 UTC 2025 🚀


### Automated Update - Thu Mar  6 00:41:02 UTC 2025 🚀


### Automated Update - Thu Mar  6 12:15:33 UTC 2025 🚀


### Automated Update - Fri Mar  7 00:41:28 UTC 2025 🚀


### Automated Update - Fri Mar  7 12:15:13 UTC 2025 🚀


### Automated Update - Sat Mar  8 00:32:36 UTC 2025 🚀


### Automated Update - Sat Mar  8 12:11:18 UTC 2025 🚀


### Automated Update - Sun Mar  9 00:36:27 UTC 2025 🚀


### Automated Update - Sun Mar  9 12:11:11 UTC 2025 🚀


### Automated Update - Mon Mar 10 00:35:31 UTC 2025 🚀


### Automated Update - Mon Mar 10 12:16:16 UTC 2025 🚀


### Automated Update - Tue Mar 11 00:41:18 UTC 2025 🚀


### Automated Update - Tue Mar 11 12:16:23 UTC 2025 🚀


### Automated Update - Wed Mar 12 00:41:00 UTC 2025 🚀


### Automated Update - Wed Mar 12 12:15:49 UTC 2025 🚀


### Automated Update - Thu Mar 13 00:41:44 UTC 2025 🚀


### Automated Update - Thu Mar 13 12:16:11 UTC 2025 🚀


### Automated Update - Fri Mar 14 00:40:55 UTC 2025 🚀


### Automated Update - Fri Mar 14 12:15:34 UTC 2025 🚀


### Automated Update - Sat Mar 15 00:40:40 UTC 2025 🚀


### Automated Update - Sat Mar 15 12:13:45 UTC 2025 🚀


### Automated Update - Sun Mar 16 00:45:09 UTC 2025 🚀


### Automated Update - Sun Mar 16 12:14:03 UTC 2025 🚀


### Automated Update - Mon Mar 17 00:43:25 UTC 2025 🚀


### Automated Update - Mon Mar 17 12:16:25 UTC 2025 🚀


### Automated Update - Tue Mar 18 00:41:30 UTC 2025 🚀


### Automated Update - Tue Mar 18 12:16:14 UTC 2025 🚀


### Automated Update - Wed Mar 19 00:41:56 UTC 2025 🚀


### Automated Update - Wed Mar 19 12:16:00 UTC 2025 🚀


### Automated Update - Thu Mar 20 00:41:13 UTC 2025 🚀


### Automated Update - Thu Mar 20 12:16:42 UTC 2025 🚀


### Automated Update - Fri Mar 21 00:42:12 UTC 2025 🚀


### Automated Update - Fri Mar 21 12:15:43 UTC 2025 🚀


### Automated Update - Sat Mar 22 00:40:53 UTC 2025 🚀


### Automated Update - Sat Mar 22 12:13:48 UTC 2025 🚀


### Automated Update - Sun Mar 23 00:45:32 UTC 2025 🚀


### Automated Update - Sun Mar 23 12:14:26 UTC 2025 🚀


### Automated Update - Mon Mar 24 00:43:47 UTC 2025 🚀


### Automated Update - Mon Mar 24 12:17:03 UTC 2025 🚀


### Automated Update - Tue Mar 25 00:42:30 UTC 2025 🚀


### Automated Update - Tue Mar 25 12:16:28 UTC 2025 🚀


### Automated Update - Wed Mar 26 00:42:10 UTC 2025 🚀


### Automated Update - Wed Mar 26 12:16:19 UTC 2025 🚀


### Automated Update - Thu Mar 27 00:42:13 UTC 2025 🚀


### Automated Update - Thu Mar 27 12:16:42 UTC 2025 🚀


### Automated Update - Fri Mar 28 00:42:06 UTC 2025 🚀


### Automated Update - Fri Mar 28 12:15:54 UTC 2025 🚀


### Automated Update - Sat Mar 29 00:41:36 UTC 2025 🚀


### Automated Update - Sat Mar 29 12:14:13 UTC 2025 🚀


### Automated Update - Sun Mar 30 00:46:16 UTC 2025 🚀


### Automated Update - Sun Mar 30 12:14:37 UTC 2025 🚀


### Automated Update - Mon Mar 31 00:45:20 UTC 2025 🚀


### Automated Update - Mon Mar 31 12:16:49 UTC 2025 🚀


### Automated Update - Tue Apr  1 00:49:52 UTC 2025 🚀


### Automated Update - Tue Apr  1 12:17:02 UTC 2025 🚀


### Automated Update - Wed Apr  2 00:42:57 UTC 2025 🚀


### Automated Update - Wed Apr  2 12:16:32 UTC 2025 🚀


### Automated Update - Thu Apr  3 00:42:13 UTC 2025 🚀


### Automated Update - Thu Apr  3 12:16:25 UTC 2025 🚀


### Automated Update - Fri Apr  4 00:42:09 UTC 2025 🚀


### Automated Update - Fri Apr  4 12:16:13 UTC 2025 🚀


### Automated Update - Sat Apr  5 00:41:37 UTC 2025 🚀


### Automated Update - Sat Apr  5 12:14:27 UTC 2025 🚀


### Automated Update - Sun Apr  6 00:46:02 UTC 2025 🚀


### Automated Update - Sun Apr  6 12:14:23 UTC 2025 🚀


### Automated Update - Mon Apr  7 00:44:23 UTC 2025 🚀


### Automated Update - Mon Apr  7 12:16:59 UTC 2025 🚀


### Automated Update - Tue Apr  8 00:42:16 UTC 2025 🚀


### Automated Update - Tue Apr  8 12:16:49 UTC 2025 🚀


### Automated Update - Wed Apr  9 01:27:23 UTC 2025 🚀


### Automated Update - Wed Apr  9 12:16:22 UTC 2025 🚀


### Automated Update - Thu Apr 10 00:42:37 UTC 2025 🚀


### Automated Update - Thu Apr 10 12:16:45 UTC 2025 🚀


### Automated Update - Fri Apr 11 00:43:22 UTC 2025 🚀


### Automated Update - Fri Apr 11 12:16:37 UTC 2025 🚀


### Automated Update - Sat Apr 12 00:42:08 UTC 2025 🚀


### Automated Update - Sat Apr 12 12:14:16 UTC 2025 🚀


### Automated Update - Sun Apr 13 02:09:38 UTC 2025 🚀


### Automated Update - Sun Apr 13 12:14:50 UTC 2025 🚀


### Automated Update - Mon Apr 14 00:45:47 UTC 2025 🚀


### Automated Update - Mon Apr 14 12:16:34 UTC 2025 🚀


### Automated Update - Tue Apr 15 00:44:03 UTC 2025 🚀


### Automated Update - Tue Apr 15 12:16:50 UTC 2025 🚀


### Automated Update - Wed Apr 16 00:44:18 UTC 2025 🚀


### Automated Update - Wed Apr 16 12:16:53 UTC 2025 🚀


### Automated Update - Thu Apr 17 00:43:07 UTC 2025 🚀


### Automated Update - Thu Apr 17 12:16:43 UTC 2025 🚀


### Automated Update - Fri Apr 18 00:42:51 UTC 2025 🚀


### Automated Update - Fri Apr 18 12:15:51 UTC 2025 🚀


### Automated Update - Sat Apr 19 00:41:29 UTC 2025 🚀


### Automated Update - Sat Apr 19 12:14:26 UTC 2025 🚀


### Automated Update - Sun Apr 20 00:47:46 UTC 2025 🚀


### Automated Update - Sun Apr 20 12:14:31 UTC 2025 🚀


### Automated Update - Mon Apr 21 00:46:29 UTC 2025 🚀


### Automated Update - Mon Apr 21 12:16:30 UTC 2025 🚀


### Automated Update - Tue Apr 22 00:43:58 UTC 2025 🚀


### Automated Update - Tue Apr 22 12:16:48 UTC 2025 🚀


### Automated Update - Wed Apr 23 00:43:34 UTC 2025 🚀


### Automated Update - Wed Apr 23 12:16:57 UTC 2025 🚀


### Automated Update - Thu Apr 24 00:43:45 UTC 2025 🚀


### Automated Update - Thu Apr 24 12:17:33 UTC 2025 🚀


### Automated Update - Fri Apr 25 00:44:11 UTC 2025 🚀


### Automated Update - Fri Apr 25 12:16:49 UTC 2025 🚀


### Automated Update - Sat Apr 26 00:42:47 UTC 2025 🚀


### Automated Update - Sat Apr 26 12:14:32 UTC 2025 🚀


### Automated Update - Sun Apr 27 00:47:46 UTC 2025 🚀


### Automated Update - Sun Apr 27 12:14:39 UTC 2025 🚀


### Automated Update - Mon Apr 28 00:46:03 UTC 2025 🚀


### Automated Update - Mon Apr 28 12:16:58 UTC 2025 🚀


### Automated Update - Tue Apr 29 00:43:58 UTC 2025 🚀


### Automated Update - Tue Apr 29 12:18:10 UTC 2025 🚀


### Automated Update - Wed Apr 30 00:44:40 UTC 2025 🚀


### Automated Update - Wed Apr 30 12:16:32 UTC 2025 🚀


### Automated Update - Thu May  1 00:50:59 UTC 2025 🚀


### Automated Update - Thu May  1 12:16:24 UTC 2025 🚀


### Automated Update - Fri May  2 00:44:34 UTC 2025 🚀


### Automated Update - Fri May  2 12:16:55 UTC 2025 🚀


### Automated Update - Sat May  3 00:43:19 UTC 2025 🚀


### Automated Update - Sat May  3 12:14:45 UTC 2025 🚀


### Automated Update - Sun May  4 00:50:53 UTC 2025 🚀


### Automated Update - Sun May  4 12:15:06 UTC 2025 🚀


### Automated Update - Mon May  5 00:48:12 UTC 2025 🚀


### Automated Update - Mon May  5 12:17:08 UTC 2025 🚀


### Automated Update - Tue May  6 00:44:44 UTC 2025 🚀


### Automated Update - Tue May  6 12:19:07 UTC 2025 🚀


### Automated Update - Wed May  7 00:44:52 UTC 2025 🚀


### Automated Update - Wed May  7 12:17:59 UTC 2025 🚀


### Automated Update - Thu May  8 00:45:18 UTC 2025 🚀


### Automated Update - Thu May  8 12:17:13 UTC 2025 🚀


### Automated Update - Fri May  9 00:44:55 UTC 2025 🚀


### Automated Update - Fri May  9 12:16:45 UTC 2025 🚀


### Automated Update - Sat May 10 00:42:55 UTC 2025 🚀


### Automated Update - Sat May 10 12:14:44 UTC 2025 🚀


### Automated Update - Sun May 11 00:49:35 UTC 2025 🚀


### Automated Update - Sun May 11 12:14:58 UTC 2025 🚀


### Automated Update - Mon May 12 00:48:36 UTC 2025 🚀


### Automated Update - Mon May 12 12:17:52 UTC 2025 🚀


### Automated Update - Tue May 13 00:45:28 UTC 2025 🚀


### Automated Update - Tue May 13 12:18:20 UTC 2025 🚀


### Automated Update - Wed May 14 00:45:14 UTC 2025 🚀


### Automated Update - Wed May 14 12:17:12 UTC 2025 🚀


### Automated Update - Thu May 15 00:44:25 UTC 2025 🚀


### Automated Update - Thu May 15 12:17:43 UTC 2025 🚀


### Automated Update - Fri May 16 00:46:11 UTC 2025 🚀


### Automated Update - Fri May 16 12:18:02 UTC 2025 🚀


### Automated Update - Sat May 17 00:44:26 UTC 2025 🚀


### Automated Update - Sat May 17 12:15:11 UTC 2025 🚀


### Automated Update - Sun May 18 00:50:30 UTC 2025 🚀


### Automated Update - Sun May 18 12:15:23 UTC 2025 🚀


### Automated Update - Mon May 19 00:49:24 UTC 2025 🚀


### Automated Update - Mon May 19 12:18:07 UTC 2025 🚀


### Automated Update - Tue May 20 00:46:57 UTC 2025 🚀


### Automated Update - Tue May 20 12:18:11 UTC 2025 🚀


### Automated Update - Wed May 21 00:46:16 UTC 2025 🚀


### Automated Update - Wed May 21 12:17:53 UTC 2025 🚀


### Automated Update - Thu May 22 00:45:13 UTC 2025 🚀


### Automated Update - Thu May 22 12:18:36 UTC 2025 🚀


### Automated Update - Fri May 23 00:45:31 UTC 2025 🚀


### Automated Update - Fri May 23 12:17:18 UTC 2025 🚀


### Automated Update - Sat May 24 00:43:31 UTC 2025 🚀


### Automated Update - Sat May 24 12:15:14 UTC 2025 🚀


### Automated Update - Sun May 25 00:51:51 UTC 2025 🚀


### Automated Update - Sun May 25 12:15:15 UTC 2025 🚀


### Automated Update - Mon May 26 00:48:00 UTC 2025 🚀


### Automated Update - Mon May 26 12:17:04 UTC 2025 🚀


### Automated Update - Tue May 27 00:44:53 UTC 2025 🚀


### Automated Update - Tue May 27 12:18:06 UTC 2025 🚀


### Automated Update - Wed May 28 00:46:07 UTC 2025 🚀


### Automated Update - Wed May 28 12:18:02 UTC 2025 🚀


### Automated Update - Thu May 29 00:46:11 UTC 2025 🚀


### Automated Update - Thu May 29 12:17:37 UTC 2025 🚀


### Automated Update - Fri May 30 00:45:36 UTC 2025 🚀


### Automated Update - Fri May 30 12:17:12 UTC 2025 🚀


### Automated Update - Sat May 31 00:44:23 UTC 2025 🚀


### Automated Update - Sat May 31 12:15:18 UTC 2025 🚀


### Automated Update - Sun Jun  1 00:57:48 UTC 2025 🚀


### Automated Update - Sun Jun  1 12:15:57 UTC 2025 🚀


### Automated Update - Mon Jun  2 00:49:38 UTC 2025 🚀


### Automated Update - Mon Jun  2 12:17:51 UTC 2025 🚀


### Automated Update - Tue Jun  3 00:47:29 UTC 2025 🚀


### Automated Update - Tue Jun  3 12:18:13 UTC 2025 🚀


### Automated Update - Wed Jun  4 00:47:01 UTC 2025 🚀


### Automated Update - Wed Jun  4 12:18:09 UTC 2025 🚀


### Automated Update - Thu Jun  5 00:46:21 UTC 2025 🚀


### Automated Update - Thu Jun  5 12:18:25 UTC 2025 🚀


### Automated Update - Fri Jun  6 00:45:38 UTC 2025 🚀


### Automated Update - Fri Jun  6 12:17:31 UTC 2025 🚀


### Automated Update - Sat Jun  7 00:45:46 UTC 2025 🚀


### Automated Update - Sat Jun  7 12:15:32 UTC 2025 🚀


### Automated Update - Sun Jun  8 00:52:42 UTC 2025 🚀


### Automated Update - Sun Jun  8 12:15:31 UTC 2025 🚀


### Automated Update - Mon Jun  9 00:51:04 UTC 2025 🚀


### Automated Update - Mon Jun  9 12:18:08 UTC 2025 🚀


### Automated Update - Tue Jun 10 00:46:51 UTC 2025 🚀


### Automated Update - Tue Jun 10 12:18:55 UTC 2025 🚀


### Automated Update - Wed Jun 11 00:47:00 UTC 2025 🚀


### Automated Update - Wed Jun 11 12:18:28 UTC 2025 🚀


### Automated Update - Thu Jun 12 00:46:28 UTC 2025 🚀


### Automated Update - Thu Jun 12 12:17:57 UTC 2025 🚀


### Automated Update - Fri Jun 13 00:47:06 UTC 2025 🚀


### Automated Update - Fri Jun 13 12:18:05 UTC 2025 🚀


### Automated Update - Sat Jun 14 00:44:58 UTC 2025 🚀


### Automated Update - Sat Jun 14 12:15:23 UTC 2025 🚀


### Automated Update - Sun Jun 15 00:53:22 UTC 2025 🚀


### Automated Update - Sun Jun 15 12:15:54 UTC 2025 🚀


### Automated Update - Mon Jun 16 00:50:01 UTC 2025 🚀


### Automated Update - Mon Jun 16 12:18:31 UTC 2025 🚀


### Automated Update - Tue Jun 17 00:47:15 UTC 2025 🚀


### Automated Update - Tue Jun 17 12:18:49 UTC 2025 🚀


### Automated Update - Wed Jun 18 00:47:09 UTC 2025 🚀


### Automated Update - Wed Jun 18 12:18:30 UTC 2025 🚀


### Automated Update - Thu Jun 19 00:47:39 UTC 2025 🚀


### Automated Update - Thu Jun 19 12:18:29 UTC 2025 🚀


### Automated Update - Fri Jun 20 00:47:09 UTC 2025 🚀


### Automated Update - Fri Jun 20 12:17:58 UTC 2025 🚀


### Automated Update - Sat Jun 21 00:45:53 UTC 2025 🚀


### Automated Update - Sat Jun 21 12:15:14 UTC 2025 🚀


### Automated Update - Sun Jun 22 00:53:12 UTC 2025 🚀


### Automated Update - Sun Jun 22 12:15:39 UTC 2025 🚀


### Automated Update - Mon Jun 23 00:51:54 UTC 2025 🚀


### Automated Update - Mon Jun 23 12:19:15 UTC 2025 🚀


### Automated Update - Tue Jun 24 00:47:51 UTC 2025 🚀


### Automated Update - Tue Jun 24 12:18:22 UTC 2025 🚀


### Automated Update - Wed Jun 25 00:48:33 UTC 2025 🚀


### Automated Update - Wed Jun 25 12:18:30 UTC 2025 🚀


### Automated Update - Thu Jun 26 00:47:43 UTC 2025 🚀


### Automated Update - Thu Jun 26 12:18:09 UTC 2025 🚀


### Automated Update - Fri Jun 27 00:48:42 UTC 2025 🚀


### Automated Update - Fri Jun 27 12:18:00 UTC 2025 🚀


### Automated Update - Sat Jun 28 00:45:39 UTC 2025 🚀


### Automated Update - Sat Jun 28 12:15:49 UTC 2025 🚀


### Automated Update - Sun Jun 29 00:54:20 UTC 2025 🚀


### Automated Update - Sun Jun 29 12:16:01 UTC 2025 🚀


### Automated Update - Mon Jun 30 00:52:13 UTC 2025 🚀


### Automated Update - Mon Jun 30 12:18:18 UTC 2025 🚀


### Automated Update - Tue Jul  1 00:54:52 UTC 2025 🚀


### Automated Update - Tue Jul  1 12:18:29 UTC 2025 🚀


### Automated Update - Wed Jul  2 00:48:02 UTC 2025 🚀


### Automated Update - Wed Jul  2 12:18:09 UTC 2025 🚀


### Automated Update - Thu Jul  3 00:47:35 UTC 2025 🚀


### Automated Update - Thu Jul  3 12:18:23 UTC 2025 🚀


### Automated Update - Fri Jul  4 00:47:33 UTC 2025 🚀


### Automated Update - Fri Jul  4 12:17:55 UTC 2025 🚀


### Automated Update - Sat Jul  5 00:45:02 UTC 2025 🚀


### Automated Update - Sat Jul  5 12:15:45 UTC 2025 🚀


### Automated Update - Sun Jul  6 00:53:36 UTC 2025 🚀


### Automated Update - Sun Jul  6 12:16:13 UTC 2025 🚀


### Automated Update - Mon Jul  7 00:52:36 UTC 2025 🚀


### Automated Update - Mon Jul  7 12:18:21 UTC 2025 🚀


### Automated Update - Tue Jul  8 00:47:56 UTC 2025 🚀


### Automated Update - Tue Jul  8 12:18:55 UTC 2025 🚀


### Automated Update - Wed Jul  9 00:49:38 UTC 2025 🚀


### Automated Update - Wed Jul  9 12:18:39 UTC 2025 🚀


### Automated Update - Thu Jul 10 00:48:53 UTC 2025 🚀


### Automated Update - Thu Jul 10 12:18:40 UTC 2025 🚀


### Automated Update - Fri Jul 11 00:49:56 UTC 2025 🚀


### Automated Update - Fri Jul 11 12:18:02 UTC 2025 🚀


### Automated Update - Sat Jul 12 00:51:06 UTC 2025 🚀


### Automated Update - Sat Jul 12 12:16:17 UTC 2025 🚀


### Automated Update - Sun Jul 13 00:55:20 UTC 2025 🚀


### Automated Update - Sun Jul 13 12:16:42 UTC 2025 🚀


### Automated Update - Mon Jul 14 00:53:07 UTC 2025 🚀


### Automated Update - Mon Jul 14 12:19:00 UTC 2025 🚀


### Automated Update - Tue Jul 15 00:51:58 UTC 2025 🚀


### Automated Update - Tue Jul 15 12:19:30 UTC 2025 🚀


### Automated Update - Wed Jul 16 00:51:00 UTC 2025 🚀


### Automated Update - Wed Jul 16 12:19:20 UTC 2025 🚀


### Automated Update - Thu Jul 17 00:51:15 UTC 2025 🚀


### Automated Update - Thu Jul 17 12:19:07 UTC 2025 🚀


### Automated Update - Fri Jul 18 00:50:34 UTC 2025 🚀


### Automated Update - Fri Jul 18 12:19:32 UTC 2025 🚀


### Automated Update - Sat Jul 19 00:49:01 UTC 2025 🚀


### Automated Update - Sat Jul 19 12:16:37 UTC 2025 🚀


### Automated Update - Sun Jul 20 00:56:35 UTC 2025 🚀


### Automated Update - Sun Jul 20 12:17:11 UTC 2025 🚀


### Automated Update - Mon Jul 21 00:54:31 UTC 2025 🚀


### Automated Update - Mon Jul 21 12:19:57 UTC 2025 🚀


### Automated Update - Tue Jul 22 00:51:27 UTC 2025 🚀


### Automated Update - Tue Jul 22 12:19:30 UTC 2025 🚀


### Automated Update - Wed Jul 23 00:51:45 UTC 2025 🚀


### Automated Update - Wed Jul 23 12:19:31 UTC 2025 🚀


### Automated Update - Thu Jul 24 00:51:23 UTC 2025 🚀


### Automated Update - Thu Jul 24 12:19:40 UTC 2025 🚀


### Automated Update - Fri Jul 25 00:51:10 UTC 2025 🚀


### Automated Update - Fri Jul 25 12:18:54 UTC 2025 🚀


### Automated Update - Sat Jul 26 00:49:30 UTC 2025 🚀


### Automated Update - Sat Jul 26 12:16:49 UTC 2025 🚀


### Automated Update - Sun Jul 27 00:56:27 UTC 2025 🚀


### Automated Update - Sun Jul 27 12:17:36 UTC 2025 🚀


### Automated Update - Mon Jul 28 00:55:18 UTC 2025 🚀


### Automated Update - Mon Jul 28 12:19:41 UTC 2025 🚀


### Automated Update - Tue Jul 29 00:56:50 UTC 2025 🚀


### Automated Update - Tue Jul 29 12:20:12 UTC 2025 🚀


### Automated Update - Wed Jul 30 00:52:10 UTC 2025 🚀


### Automated Update - Wed Jul 30 12:20:11 UTC 2025 🚀


### Automated Update - Thu Jul 31 00:52:22 UTC 2025 🚀


### Automated Update - Thu Jul 31 12:18:04 UTC 2025 🚀


### Automated Update - Fri Aug  1 00:58:12 UTC 2025 🚀


### Automated Update - Fri Aug  1 12:19:23 UTC 2025 🚀


### Automated Update - Sat Aug  2 00:49:14 UTC 2025 🚀


### Automated Update - Sat Aug  2 12:17:30 UTC 2025 🚀


### Automated Update - Sun Aug  3 00:57:28 UTC 2025 🚀


### Automated Update - Sun Aug  3 12:17:39 UTC 2025 🚀


### Automated Update - Mon Aug  4 00:57:00 UTC 2025 🚀


### Automated Update - Mon Aug  4 12:20:17 UTC 2025 🚀


### Automated Update - Tue Aug  5 00:53:29 UTC 2025 🚀


### Automated Update - Tue Aug  5 12:20:47 UTC 2025 🚀


### Automated Update - Wed Aug  6 00:52:51 UTC 2025 🚀


### Automated Update - Wed Aug  6 12:20:43 UTC 2025 🚀


### Automated Update - Thu Aug  7 00:53:06 UTC 2025 🚀


### Automated Update - Thu Aug  7 12:20:38 UTC 2025 🚀


### Automated Update - Fri Aug  8 00:52:39 UTC 2025 🚀


### Automated Update - Fri Aug  8 12:19:57 UTC 2025 🚀


### Automated Update - Sat Aug  9 00:46:47 UTC 2025 🚀


### Automated Update - Sat Aug  9 12:16:49 UTC 2025 🚀


### Automated Update - Sun Aug 10 00:55:37 UTC 2025 🚀


### Automated Update - Sun Aug 10 12:17:27 UTC 2025 🚀


### Automated Update - Mon Aug 11 00:53:43 UTC 2025 🚀


### Automated Update - Mon Aug 11 12:19:42 UTC 2025 🚀


### Automated Update - Tue Aug 12 00:47:24 UTC 2025 🚀


### Automated Update - Tue Aug 12 12:18:43 UTC 2025 🚀


### Automated Update - Wed Aug 13 00:48:37 UTC 2025 🚀


### Automated Update - Wed Aug 13 12:18:55 UTC 2025 🚀


### Automated Update - Thu Aug 14 00:48:40 UTC 2025 🚀


### Automated Update - Thu Aug 14 12:19:23 UTC 2025 🚀


### Automated Update - Fri Aug 15 00:49:09 UTC 2025 🚀


### Automated Update - Fri Aug 15 12:17:58 UTC 2025 🚀


### Automated Update - Sat Aug 16 00:44:50 UTC 2025 🚀


### Automated Update - Sat Aug 16 12:16:14 UTC 2025 🚀


### Automated Update - Sun Aug 17 00:53:05 UTC 2025 🚀


### Automated Update - Sun Aug 17 12:16:45 UTC 2025 🚀


### Automated Update - Mon Aug 18 00:52:44 UTC 2025 🚀


### Automated Update - Mon Aug 18 12:19:18 UTC 2025 🚀


### Automated Update - Tue Aug 19 00:46:16 UTC 2025 🚀


### Automated Update - Tue Aug 19 12:18:09 UTC 2025 🚀


### Automated Update - Wed Aug 20 00:44:13 UTC 2025 🚀


### Automated Update - Wed Aug 20 12:18:01 UTC 2025 🚀


### Automated Update - Thu Aug 21 00:43:12 UTC 2025 🚀


### Automated Update - Thu Aug 21 12:18:01 UTC 2025 🚀


### Automated Update - Fri Aug 22 00:44:54 UTC 2025 🚀


### Automated Update - Fri Aug 22 12:17:29 UTC 2025 🚀


### Automated Update - Sat Aug 23 00:42:45 UTC 2025 🚀


### Automated Update - Sat Aug 23 12:15:21 UTC 2025 🚀


### Automated Update - Sun Aug 24 00:51:23 UTC 2025 🚀


### Automated Update - Sun Aug 24 12:15:50 UTC 2025 🚀


### Automated Update - Mon Aug 25 00:47:10 UTC 2025 🚀


### Automated Update - Mon Aug 25 12:18:15 UTC 2025 🚀


### Automated Update - Tue Aug 26 00:44:51 UTC 2025 🚀


### Automated Update - Tue Aug 26 12:18:54 UTC 2025 🚀


### Automated Update - Wed Aug 27 00:44:09 UTC 2025 🚀


### Automated Update - Wed Aug 27 12:17:36 UTC 2025 🚀


### Automated Update - Thu Aug 28 00:43:27 UTC 2025 🚀


### Automated Update - Thu Aug 28 12:17:37 UTC 2025 🚀


### Automated Update - Fri Aug 29 00:44:13 UTC 2025 🚀


### Automated Update - Fri Aug 29 12:16:51 UTC 2025 🚀


### Automated Update - Sat Aug 30 00:41:09 UTC 2025 🚀


### Automated Update - Sat Aug 30 12:15:16 UTC 2025 🚀


### Automated Update - Sun Aug 31 00:47:25 UTC 2025 🚀


### Automated Update - Sun Aug 31 12:15:26 UTC 2025 🚀


### Automated Update - Mon Sep  1 00:54:22 UTC 2025 🚀


### Automated Update - Mon Sep  1 12:18:14 UTC 2025 🚀


### Automated Update - Tue Sep  2 00:43:55 UTC 2025 🚀


### Automated Update - Tue Sep  2 12:17:58 UTC 2025 🚀


### Automated Update - Wed Sep  3 00:41:02 UTC 2025 🚀


### Automated Update - Wed Sep  3 12:17:29 UTC 2025 🚀


### Automated Update - Thu Sep  4 00:41:37 UTC 2025 🚀


### Automated Update - Thu Sep  4 12:17:28 UTC 2025 🚀


### Automated Update - Fri Sep  5 00:42:30 UTC 2025 🚀


### Automated Update - Fri Sep  5 12:16:30 UTC 2025 🚀


### Automated Update - Sat Sep  6 00:41:07 UTC 2025 🚀


### Automated Update - Sat Sep  6 12:14:37 UTC 2025 🚀


### Automated Update - Sun Sep  7 00:47:06 UTC 2025 🚀


### Automated Update - Sun Sep  7 12:15:05 UTC 2025 🚀


### Automated Update - Mon Sep  8 00:45:32 UTC 2025 🚀


### Automated Update - Mon Sep  8 12:18:53 UTC 2025 🚀


### Automated Update - Tue Sep  9 00:42:55 UTC 2025 🚀


### Automated Update - Tue Sep  9 12:19:04 UTC 2025 🚀


### Automated Update - Wed Sep 10 00:42:10 UTC 2025 🚀


### Automated Update - Wed Sep 10 12:17:18 UTC 2025 🚀


### Automated Update - Thu Sep 11 00:42:43 UTC 2025 🚀


### Automated Update - Thu Sep 11 12:16:56 UTC 2025 🚀


### Automated Update - Fri Sep 12 00:41:23 UTC 2025 🚀


### Automated Update - Fri Sep 12 12:17:26 UTC 2025 🚀


### Automated Update - Sat Sep 13 00:39:36 UTC 2025 🚀


### Automated Update - Sat Sep 13 12:14:41 UTC 2025 🚀


### Automated Update - Sun Sep 14 00:45:39 UTC 2025 🚀


### Automated Update - Sun Sep 14 12:14:47 UTC 2025 🚀


### Automated Update - Mon Sep 15 00:46:04 UTC 2025 🚀


### Automated Update - Mon Sep 15 12:17:44 UTC 2025 🚀


### Automated Update - Tue Sep 16 00:41:41 UTC 2025 🚀


### Automated Update - Tue Sep 16 12:17:29 UTC 2025 🚀


### Automated Update - Wed Sep 17 00:41:55 UTC 2025 🚀


### Automated Update - Wed Sep 17 12:17:40 UTC 2025 🚀


### Automated Update - Thu Sep 18 00:41:15 UTC 2025 🚀


### Automated Update - Thu Sep 18 12:17:01 UTC 2025 🚀


### Automated Update - Fri Sep 19 00:43:28 UTC 2025 🚀


### Automated Update - Fri Sep 19 12:17:35 UTC 2025 🚀


### Automated Update - Sat Sep 20 00:40:48 UTC 2025 🚀


### Automated Update - Sat Sep 20 12:15:32 UTC 2025 🚀


### Automated Update - Sun Sep 21 00:47:30 UTC 2025 🚀


### Automated Update - Sun Sep 21 12:15:13 UTC 2025 🚀


### Automated Update - Mon Sep 22 00:46:36 UTC 2025 🚀


### Automated Update - Mon Sep 22 12:18:10 UTC 2025 🚀


### Automated Update - Tue Sep 23 00:42:17 UTC 2025 🚀


### Automated Update - Tue Sep 23 12:17:21 UTC 2025 🚀


### Automated Update - Wed Sep 24 00:42:56 UTC 2025 🚀


### Automated Update - Wed Sep 24 12:18:03 UTC 2025 🚀


### Automated Update - Thu Sep 25 00:42:59 UTC 2025 🚀


### Automated Update - Thu Sep 25 12:18:20 UTC 2025 🚀


### Automated Update - Fri Sep 26 00:42:23 UTC 2025 🚀


### Automated Update - Fri Sep 26 12:17:34 UTC 2025 🚀


### Automated Update - Sat Sep 27 00:41:00 UTC 2025 🚀


### Automated Update - Sat Sep 27 12:15:07 UTC 2025 🚀


### Automated Update - Sun Sep 28 00:48:00 UTC 2025 🚀


### Automated Update - Sun Sep 28 12:15:10 UTC 2025 🚀


### Automated Update - Mon Sep 29 00:44:29 UTC 2025 🚀


### Automated Update - Mon Sep 29 12:18:22 UTC 2025 🚀


### Automated Update - Tue Sep 30 00:43:17 UTC 2025 🚀


### Automated Update - Tue Sep 30 12:18:22 UTC 2025 🚀


### Automated Update - Wed Oct  1 00:50:04 UTC 2025 🚀


### Automated Update - Wed Oct  1 12:18:23 UTC 2025 🚀


### Automated Update - Thu Oct  2 00:41:42 UTC 2025 🚀


### Automated Update - Thu Oct  2 12:16:40 UTC 2025 🚀


### Automated Update - Fri Oct  3 00:41:49 UTC 2025 🚀


### Automated Update - Fri Oct  3 12:16:46 UTC 2025 🚀


### Automated Update - Sat Oct  4 00:39:36 UTC 2025 🚀


### Automated Update - Sat Oct  4 12:14:38 UTC 2025 🚀


### Automated Update - Sun Oct  5 00:47:22 UTC 2025 🚀


### Automated Update - Sun Oct  5 12:14:54 UTC 2025 🚀


### Automated Update - Mon Oct  6 00:43:41 UTC 2025 🚀


### Automated Update - Mon Oct  6 12:17:47 UTC 2025 🚀


### Automated Update - Tue Oct  7 00:42:23 UTC 2025 🚀


### Automated Update - Tue Oct  7 12:18:24 UTC 2025 🚀


### Automated Update - Wed Oct  8 00:42:02 UTC 2025 🚀


### Automated Update - Wed Oct  8 12:18:34 UTC 2025 🚀


### Automated Update - Thu Oct  9 00:42:56 UTC 2025 🚀


### Automated Update - Thu Oct  9 12:17:45 UTC 2025 🚀


### Automated Update - Fri Oct 10 00:42:32 UTC 2025 🚀


### Automated Update - Fri Oct 10 12:18:32 UTC 2025 🚀


### Automated Update - Sat Oct 11 00:40:14 UTC 2025 🚀


### Automated Update - Sat Oct 11 12:15:03 UTC 2025 🚀


### Automated Update - Sun Oct 12 00:44:56 UTC 2025 🚀


### Automated Update - Sun Oct 12 12:15:24 UTC 2025 🚀


### Automated Update - Mon Oct 13 00:46:35 UTC 2025 🚀


### Automated Update - Mon Oct 13 12:18:20 UTC 2025 🚀


### Automated Update - Tue Oct 14 00:42:34 UTC 2025 🚀


### Automated Update - Tue Oct 14 12:18:54 UTC 2025 🚀


### Automated Update - Wed Oct 15 00:44:18 UTC 2025 🚀


### Automated Update - Wed Oct 15 12:19:30 UTC 2025 🚀


### Automated Update - Thu Oct 16 00:44:05 UTC 2025 🚀


### Automated Update - Thu Oct 16 12:18:33 UTC 2025 🚀


### Automated Update - Fri Oct 17 00:43:22 UTC 2025 🚀


### Automated Update - Fri Oct 17 12:17:50 UTC 2025 🚀


### Automated Update - Sat Oct 18 00:40:54 UTC 2025 🚀


### Automated Update - Sat Oct 18 12:15:32 UTC 2025 🚀


### Automated Update - Sun Oct 19 00:49:51 UTC 2025 🚀


### Automated Update - Sun Oct 19 12:15:48 UTC 2025 🚀


### Automated Update - Mon Oct 20 00:48:22 UTC 2025 🚀


### Automated Update - Mon Oct 20 12:18:24 UTC 2025 🚀


### Automated Update - Tue Oct 21 00:44:34 UTC 2025 🚀


### Automated Update - Tue Oct 21 12:18:49 UTC 2025 🚀


### Automated Update - Wed Oct 22 00:45:42 UTC 2025 🚀


### Automated Update - Wed Oct 22 12:18:45 UTC 2025 🚀


### Automated Update - Thu Oct 23 00:44:17 UTC 2025 🚀


### Automated Update - Thu Oct 23 12:18:46 UTC 2025 🚀


### Automated Update - Fri Oct 24 00:41:26 UTC 2025 🚀


### Automated Update - Fri Oct 24 12:18:48 UTC 2025 🚀


### Automated Update - Sat Oct 25 00:42:58 UTC 2025 🚀


### Automated Update - Sat Oct 25 12:15:24 UTC 2025 🚀


### Automated Update - Sun Oct 26 00:47:56 UTC 2025 🚀


### Automated Update - Sun Oct 26 12:16:04 UTC 2025 🚀


### Automated Update - Mon Oct 27 00:50:18 UTC 2025 🚀


### Automated Update - Mon Oct 27 12:18:48 UTC 2025 🚀


### Automated Update - Tue Oct 28 00:43:16 UTC 2025 🚀


### Automated Update - Tue Oct 28 12:18:12 UTC 2025 🚀


### Automated Update - Wed Oct 29 00:46:57 UTC 2025 🚀


### Automated Update - Wed Oct 29 12:18:59 UTC 2025 🚀


### Automated Update - Thu Oct 30 00:46:33 UTC 2025 🚀


### Automated Update - Thu Oct 30 12:18:33 UTC 2025 🚀


### Automated Update - Fri Oct 31 00:44:32 UTC 2025 🚀


### Automated Update - Fri Oct 31 12:18:53 UTC 2025 🚀


### Automated Update - Sat Nov  1 00:48:50 UTC 2025 🚀


### Automated Update - Sat Nov  1 12:15:35 UTC 2025 🚀


### Automated Update - Sun Nov  2 00:49:32 UTC 2025 🚀


### Automated Update - Sun Nov  2 12:15:37 UTC 2025 🚀


### Automated Update - Mon Nov  3 00:48:46 UTC 2025 🚀


### Automated Update - Mon Nov  3 12:18:52 UTC 2025 🚀


### Automated Update - Tue Nov  4 00:45:06 UTC 2025 🚀


### Automated Update - Tue Nov  4 12:19:39 UTC 2025 🚀


### Automated Update - Wed Nov  5 00:47:15 UTC 2025 🚀


### Automated Update - Wed Nov  5 12:18:45 UTC 2025 🚀


### Automated Update - Thu Nov  6 00:45:45 UTC 2025 🚀


### Automated Update - Thu Nov  6 12:18:43 UTC 2025 🚀


### Automated Update - Fri Nov  7 00:45:42 UTC 2025 🚀


### Automated Update - Fri Nov  7 12:18:02 UTC 2025 🚀


### Automated Update - Sat Nov  8 00:43:05 UTC 2025 🚀


### Automated Update - Sat Nov  8 12:16:08 UTC 2025 🚀


### Automated Update - Sun Nov  9 00:49:17 UTC 2025 🚀


### Automated Update - Sun Nov  9 12:15:41 UTC 2025 🚀


### Automated Update - Mon Nov 10 00:49:15 UTC 2025 🚀


### Automated Update - Mon Nov 10 12:18:48 UTC 2025 🚀


### Automated Update - Tue Nov 11 00:47:05 UTC 2025 🚀


### Automated Update - Tue Nov 11 12:18:35 UTC 2025 🚀


### Automated Update - Wed Nov 12 00:46:52 UTC 2025 🚀


### Automated Update - Wed Nov 12 12:19:07 UTC 2025 🚀


### Automated Update - Thu Nov 13 00:46:37 UTC 2025 🚀


### Automated Update - Thu Nov 13 12:19:07 UTC 2025 🚀


### Automated Update - Fri Nov 14 00:46:30 UTC 2025 🚀


### Automated Update - Fri Nov 14 12:19:05 UTC 2025 🚀


### Automated Update - Sat Nov 15 00:44:46 UTC 2025 🚀


### Automated Update - Sat Nov 15 12:16:07 UTC 2025 🚀


### Automated Update - Sun Nov 16 00:50:39 UTC 2025 🚀


### Automated Update - Sun Nov 16 12:16:21 UTC 2025 🚀


### Automated Update - Mon Nov 17 00:48:16 UTC 2025 🚀


### Automated Update - Mon Nov 17 12:18:55 UTC 2025 🚀


### Automated Update - Tue Nov 18 00:45:28 UTC 2025 🚀


### Automated Update - Tue Nov 18 12:19:25 UTC 2025 🚀


### Automated Update - Wed Nov 19 00:46:19 UTC 2025 🚀


### Automated Update - Wed Nov 19 12:18:51 UTC 2025 🚀


### Automated Update - Thu Nov 20 00:44:58 UTC 2025 🚀


### Automated Update - Thu Nov 20 12:18:55 UTC 2025 🚀


### Automated Update - Fri Nov 21 00:45:30 UTC 2025 🚀


### Automated Update - Fri Nov 21 12:17:57 UTC 2025 🚀


### Automated Update - Sat Nov 22 00:43:56 UTC 2025 🚀


### Automated Update - Sat Nov 22 12:15:53 UTC 2025 🚀


### Automated Update - Sun Nov 23 00:53:57 UTC 2025 🚀


### Automated Update - Sun Nov 23 12:15:28 UTC 2025 🚀


### Automated Update - Mon Nov 24 00:50:46 UTC 2025 🚀


### Automated Update - Mon Nov 24 12:19:08 UTC 2025 🚀
