# Abstractive Visual Understanding of Multi-modal Structured Knowledge: A New Perspective for MLLM Evaluation

> Multi-modal large language models (MLLMs) incorporate heterogeneous modalities into LLMs, enabling a comprehensive understanding of diverse scenarios and objects. Despite the proliferation of evaluation benchmarks and leaderboards for MLLMs, they predominantly overlook the critical capacity of MLLMs to comprehend world knowledge with structured abstractions that appear in visual form. To address this gap, we propose a novel evaluation paradigm and devise M3STR, an innovative benchmark grounded in the Multi-Modal Map for STRuctured understanding. This benchmark leverages multi-modal knowledge graphs to synthesize images encapsulating subgraph architectures enriched with multi-modal entities. M3STR necessitates that MLLMs not only recognize the multi-modal entities within the visual inputs but also decipher intricate relational topologies among them. We delineate the benchmark's statistical profiles and automated construction pipeline, accompanied by an extensive empirical analysis of 26 state-of-the-art MLLMs. Our findings reveal persistent deficiencies in processing abstractive visual information with structured knowledge, thereby charting a pivotal trajectory for advancing MLLMs' holistic reasoning capacities.



## Overview
![model](resource/key_vision.png)

## 🎆 News

- `2025-06` We released our pre-print paper [Abstractive Visual Understanding of Multi-modal Structured Knowledge: A New Perspective for MLLM Evaluation](https://arxiv.org/abs/2506.01293) on ArXiV.



## 📏 Dependencies and Supported MLLMs

### Supported MLLMs
- LLaVA: LLaVA-1.5-7B, LLaVA-1.6-vicuna-7B, LLAVA-llama-3-8b-v1.1
- InstructBLIP: InstructBLIP-7B, InstructBLIP-13B
- Deepseek-VL: Deepseek-VL-1.3B-chat, Deepseek-VL-7B-chat, Deepseek-VL2-tiny, Deepseek-VL2-small
- Intern2.5-VL: Intern2.5-VL-1B, Intern2.5-VL-8B
- Phi-vision: Phi3-vision, Phi3.5-vision
- MiniCPM-V: MiniCPM-V-2, MiniCPM-V-2.5, MiniCPM-V-2.6
- Qwen-VL: Qwen2-VL-2B, Qwen2-VL-7B, Qwen2-VL-72B, Qwen2.5-VL-3B, Qwen2.5-VL-7B, Qwen2.5-VL-72B
- Others: Chameleon-7B

Before you run the evaluate experiments, you should donwload the MLLM weights from Huggingface Model Hub or Model Scope, and set the `model_path_map` in utils.py.

Note that our code supports several different MLLMs. However, since different MLLM runs require different python environments, we need to configure multiple environments to run evaluations for different models.
- For most of the MLLMs, you need the basic python environments in `envs/requirements_base.txt`.
- For Qwen-VL models, you need to setup a new python environment with the config in `envs/requirements_qwen.txt`.
- For Deepseek-VL/VL2, you need `envs/requirements_deepseek1.txt` and `envs/requirements_deepseek2.txt` respectively. Note that you should also follow the instructions in official reposities to install the extra libraries required for Deepseek-VL models. We have prepared them in `DeepSeek-VL/` and `DeepSeek-VL2`.




## 🌲 Data Preparation
We have prepared the images in [Google Drive](https://drive.google.com/file/d/10zvn5R6S6DmwCA9hSInAhl6Yn1ImIiGl/view?usp=sharing). Please download from this link and put the image files in `dataset/images/`.

## 🐊 Evaluation
You can run the evaluation code with the following scripts:

```shell
python run_task1_evaluation_vllm.py\
--model_type internvl \
--model_used all
```



## 🤝 Citation
```bigquery
@misc{zhang2025abstractivevisualunderstandingmultimodal,
      title={Abstractive Visual Understanding of Multi-modal Structured Knowledge: A New Perspective for MLLM Evaluation}, 
      author={Yichi Zhang and Zhuo Chen and Lingbing Guo and Yajing Xu and Min Zhang and Wen Zhang and Huajun Chen},
      year={2025},
      eprint={2506.01293},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.01293}, 
}
```