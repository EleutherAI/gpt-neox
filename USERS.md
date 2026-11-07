# Models and Publications Using GPT-NeoX

This file lists models and publications that used GPT-NeoX. The list includes more than 50 models and more than 60 publications from more than 40 organizations.

These models were trained on NVIDIA and AMD GPUs. The systems used include ORNL Frontier and Summit, LUMI, CINECA Leonardo, AWS, and CoreWeave.

Each entry links to a paper, model card, or repository that states GPT-NeoX was used. Some entries used a training framework derived from GPT-NeoX, and those entries say so.

If you have used GPT-NeoX in your work, please open a pull request or contact us so we can add it.

## Publications

The following publications use GPT-NeoX or a framework derived from it. They are grouped by topic. Authors affiliated with EleutherAI are shown in **bold**.

### Pretraining, Data, and Model Development

- **Black**, **Biderman**, **Hallahan**, **Anthony**, **Gao**, **Golding**, **He**, **Leahy**, **McDonell**, **Phang**, **Pieler**, **Purohit**, **Reynolds**, **Tow**, **Wang**, and **Weinbach**. "[GPT-NeoX-20B: An Open-Source Autoregressive Language Model](https://aclanthology.org/2022.bigscience-1.9/)." In _Proceedings of BigScience Episode #5 — Workshop on Challenges & Perspectives in Creating Large Language Models_, 2022.
- Albalak, Pan, Raffel, and Wang. "[Efficient Online Data Mixing For Language Model Pre-Training](https://neurips.cc/virtual/2023/78234)." In _NeurIPS Workshop on R0-FoMo: Robustness of Few-shot and Zero-shot Learning in Large Foundation Models_, 2023.
- Gu, Lyu, Li, and Zhang. "[Data Mixing Can Induce Phase Transitions in Knowledge Acquisition](https://papers.neurips.cc/paper_files/paper/2025/hash/eee7ae5cf0c4356c2aeca400771791aa-Abstract-Conference.html)." In _Advances in Neural Information Processing Systems 38_, 2025.
- Khan, Godbole, Wei, Wang, Flemings, Gummadi, Neiswanger, and Jia. "[TokenSmith: Streamlining Data Editing, Search, and Inspection for Large-Scale Language Model Training and Interpretability](https://aclanthology.org/2025.emnlp-demos.50/)." In _Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations_, 2025.
- Dang, Ma, and Liao. "[Holistic Data Scheduler for LLM Pre-training via Multi-Objective Reinforcement Learning](https://doi.org/10.1145/3770854.3780325)." In _Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Volume 1_, pp. 176–187, 2026.

### Model Architecture and Training Methods

- Chi, Fan, Chen, Rudnicky, and Ramadge. "[Latent Positional Information is in the Self-Attention Variance of Transformer Language Models Without Positional Embeddings](https://aclanthology.org/2023.acl-short.102/)." In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)_, pp. 1183-1193, 2023.
- Shen, Zhao, Dong, Li, Li, Sun, and Zeng. "[Astrocyte-Enabled Advancements in Spiking Neural Networks for Large Language Modeling](https://arxiv.org/abs/2312.07625)." In _arXiv preprint_, 2023.
- Ho, Bae, Kim, Jo, Kim, Schuster, Fisch, Thorne, and Yun. "[Block Transformer: Global-to-Local Language Modeling for Fast Inference](https://papers.nips.cc/paper_files/paper/2024/hash/579b5b84311e122584dedab1d3b7613f-Abstract-Conference.html)." In _Advances in Neural Information Processing Systems 37_, 2024.
- Mallik, Janowski, Hog, Rakotoarison, Klein, Grabocka, and Hutter. "[Warmstarting for Scaling Language Models](https://openreview.net/forum?id=2SilklW9NA)." In _NeurIPS Workshop on Adaptive Foundation Models_, 2024.
- Qiu, Guo, Xu, Zhao, Zhang, and Yang. "[To Cool or not to Cool? Temperature Network Meets Large Foundation Models via DRO](https://proceedings.mlr.press/v235/qiu24c.html)." In _International Conference on Machine Learning_, 2024.
- Yu, Gupta, Gopalswamy, Mamidala, Zhou, Huynh, Park, Diamant, Deoras, and Huan. "[Collage: Light-Weight Low-Precision Strategy for LLM Training](https://proceedings.mlr.press/v235/yu24d.html)." In _International Conference on Machine Learning_, 2024.
- Zeng, Guo, Fei, Yin, Zhou, Li, Sun, Yan, Lin, and Qiu. "[Turn Waste into Worth: Rectifying Top-k Router of MoE](https://aclanthology.org/2024.emnlp-main.739/)." In _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing_, 2024.
- Kaushal, Vaidhya, Mondal, Pandey, Bhagat, and Rish. "[Surprising Effectiveness of Pretraining Ternary Language Model at Scale](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6499f26637f74f7c0fbfb46668434973-Abstract-Conference.html)." In _International Conference on Learning Representations_, 2025.
- Khan, Gummadi, Gupta, and Ravichander. "[Fractional Rotation, Full Potential? Investigating Performance and Convergence of Partial RoPE](https://arxiv.org/abs/2603.11611)." In _arXiv preprint arXiv:2603.11611_, 2026.
- Klein, Kusch, Sager, Schnake, and Schotthöfer. "[Tucker Attention: A Generalization of Approximate Attention Mechanisms](https://arxiv.org/abs/2603.30033)." In _International Conference on Machine Learning_, 2026.
- Zimerman, Adir, Aharoni, Avitan, Baruch, Drucker, Lerner, Masalha, Moshe, and Soceanu. "[PowerSoftmax: Towards Secure LLM Inference over Encrypted Data](https://openreview.net/forum?id=WeIUIh1xwt)." In _International Conference on Artificial Intelligence and Statistics_, 2026.

### Training Systems and HPC

- Yin, Dash, Gounley, Wang, and Tourassi. "[Evaluation of Pre-Training Large Language Models on Leadership-Class Supercomputers](https://link.springer.com/article/10.1007/s11227-023-05479-7)." In _The Journal of Supercomputing_ 79(18), pp. 20747–20768, 2023.
- Yin, Dash, Wang, and Shankar. "[FORGE: Pre-Training Open Foundation Models for Science](https://doi.org/10.1145/3581784.3613215)." In _Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '23)_, pp. 1-13, 2023. ([ORNL publication page](https://www.ornl.gov/publication/forge-pre-training-open-foundation-models-science))
- Anthony, Michalowicz, Hatef, Xu, Abduljabbar, Shafi, Subramoni, and Panda. "[Demystifying the Communication Characteristics for Distributed Transformer Models](https://doi.org/10.1109/HOTI63208.2024.00020)." In _IEEE Symposium on High-Performance Interconnects_, pp. 57–65, 2024.
- **Anthony**, **Hatef**, Narayanan, **Biderman**, Bekman, Yin, Shafi, Subramoni, and Panda. "[The Case for Co-Designing Model Architectures with Hardware](https://doi.org/10.1145/3673038.3673136)." In _Proceedings of the 53rd International Conference on Parallel Processing_, 2024.
- Patel, Choukse, Zhang, Goiri, Warrier, Mahalingam, and Bianchini. "[Characterizing Power Management Opportunities for LLMs in the Cloud](https://doi.org/10.1145/3620666.3651329)." In _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3_, pp. 207–222, 2024.
- Xu, Anthony, Zhou, Alnaasan, Gulhane, Shafi, Subramoni, and Panda. "[Accelerating Large Language Model Training with Hybrid GPU-based Compression](https://doi.org/10.1109/CCGrid59990.2024.00031)." In _IEEE/ACM International Symposium on Cluster, Cloud and Internet Computing_, pp. 196–205, 2024.
- Xu, Anthony, Hatef, Shafi, Subramoni, and Panda. "[Scaling Large Language Model Training on Frontier with Low-Bandwidth Partitioning](https://doi.org/10.1109/HIPC62374.2024.00015)." In _IEEE 31st International Conference on High Performance Computing, Data, and Analytics_, pp. 57–67, 2024.
- Yin, Bose, Cong, Lyngaas, and **Anthony**. "[Comparative Study of Large Language Model Architectures on Frontier](https://ieeexplore.ieee.org/document/10579218)." In _IEEE International Parallel and Distributed Processing Symposium_, 2024.

### Continual Pretraining and Model Adaptation

- Gupta, Thérien, Ibrahim, Richter, **Anthony**, Belilovsky, Rish, and Lesort. "[Continual Pre-Training of Large Language Models: How to re-warm your model?](https://icml.cc/virtual/2023/28270)." In _Workshop on Efficient Systems for Foundation Models @ ICML_, 2023.
- Kaddour and Liu. "[Synthetic Data Generation in Low-Resource Settings via Fine-Tuning of Large Language Models](https://arxiv.org/abs/2310.01119)." _arXiv:2310.01119_, 2023.
- Ibrahim, Thérien, Gupta, Richter, **Anthony**, Lesort, Belilovsky, and Rish. "[Simple and Scalable Strategies to Continually Pre-train Large Language Models](https://openreview.net/forum?id=DimPeeCxKO)." In _Transactions on Machine Learning Research_, 2024.
- Ji, Li, Paavola, Luo, and Tiedemann. "[Data-Centric Continual Pre-training for 500+ Languages: A New Bilingual Translation Corpus and Multilingual Models](https://aclanthology.org/2026.findings-acl.937/)." In _Findings of the Association for Computational Linguistics: ACL 2026_, pp. 18776–18807, 2026.

### Long-Context Modeling and Length Extrapolation

- Chi, Fan, Ramadge, and Rudnicky. "[KERPLE: Kernelized Relative Positional Embedding for Length Extrapolation](https://proceedings.neurips.cc/paper_files/paper/2022/hash/37a413841a614b5414b333585e7613b8-Abstract-Conference.html)." In _Advances in Neural Information Processing Systems 35_, 2022.
- Chi, Fan, Rudnicky, and Ramadge. "[Dissecting Transformer Length Extrapolation via the Lens of Receptive Field Analysis](https://aclanthology.org/2023.acl-long.756/)." In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, pp. 13522-13537, 2023.
- Gao. "[MEP: Multiple Kernel Learning Enhancing Relative Positional Encoding Length Extrapolation](https://arxiv.org/abs/2403.17698)." In _arXiv preprint arXiv:2403.17698_, 2024.
- Gao, Wu, Fu, and Hu. "[Quest: Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e54e6eef11f87a874bf1e4551fc6d04e-Abstract-Conference.html)." In _International Conference on Learning Representations_, 2025.
- Gao, Wu, Lin, Zhang, and Hu. "[NExtLong: Toward Effective Long-Context Training without Long Documents](https://proceedings.mlr.press/v267/gao25n.html)." In _International Conference on Machine Learning_, 2025.

### Interpretability, Memorization, and Alignment

- **Biderman**, **Prashanth**, **Sutawika**, **Schoelkopf**, **Anthony**, **Purohit**, and Raff. "[Emergent and Predictable Memorization in Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/59404fb89d6194641c69ae99ecdf8f6d-Abstract-Conference.html)." In _Advances in Neural Information Processing Systems 36_, 2023.
- **Biderman**, **Schoelkopf**, **Anthony**, **Bradley**, **O'Brien**, **Hallahan**, Khan, **Purohit**, **Prashanth**, Raff, **Skowron**, **Sutawika**, and **van der Wal**. "[Pythia: A suite for analyzing large language models across training and scaling](https://proceedings.mlr.press/v202/biderman23a.html)." In *International Conference on Machine Learning*, pp. 2397-2430. *PMLR*, 2023.
- Havrilla, Zhuravinskyi, Phung, Tiwari, Tow, **Biderman**, **Anthony**, and **Castricato**. "[trlX: A Framework for Large Scale Reinforcement Learning from Human Feedback](https://aclanthology.org/2023.emnlp-main.530/)." In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_, 2023.
- Wei, Wang, and Jia. "[Proving Membership in LLM Pretraining Data via Data Watermarks](https://aclanthology.org/2024.findings-acl.788/)." In _Findings of the Association for Computational Linguistics: ACL 2024_, 2024.
- Xiao, Zhou, Ping, Cao, Li, Zhou, Li, and Bogdan. "[Exploring Neuron Interactions and Emergence in LLMs: From the Multifractal Analysis Perspective](https://arxiv.org/abs/2402.09099)." In _arXiv preprint_, 2024.
- **O'Brien**, Casper, **Anthony**, Korbak, Kirk, Davies, Mishra, Irving, Gal, and **Biderman**. "[Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight LLMs](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3bf80b34f731313b8292f4578e820c90-Abstract-Conference.html)." In _International Conference on Learning Representations_, 2026.
- Tice, Radmard, Ratnam, Kim, Africa, and O'Brien. "[Alignment Pretraining: AI Discourse Causes Self-Fulfilling (Mis)alignment](https://openreview.net/forum?id=951OAanYyQ)." In _International Conference on Machine Learning_ (Spotlight), 2026.
- Wei, Godbole, Khan, Wang, Zhu, Flemings, Kashyap, Gummadi, Neiswanger, and Jia. "[Hubble: A Model Suite to Advance the Study of LLM Memorization](https://openreview.net/forum?id=ZfdnZhOP0k)." In _International Conference on Learning Representations_ (Oral), 2026.

### Language, Cognition, and Neuroscience

- Hosseini and Fedorenko. "[Large Language Models Implicitly Learn to Straighten Neural Sentence Trajectories to Construct a Predictive Representation of Natural Language](https://proceedings.neurips.cc/paper_files/paper/2023/hash/88dddaf430b5bc38ab8228902bb61821-Abstract-Conference.html)." In _Advances in Neural Information Processing Systems 36_, 2023.
- Oh and Schuler. "[Transformer-Based Language Model Surprisal Predicts Human Reading Times Best with About Two Billion Training Tokens](https://aclanthology.org/2023.findings-emnlp.128/)." In _Findings of the Association for Computational Linguistics: EMNLP 2023_, 2023.
- Hosseini, Schrimpf, Zhang, Bowman, Zaslavsky, and Fedorenko. "[Artificial Neural Network Language Models Predict Human Brain Responses to Language Even After a Developmentally Realistic Amount of Training](https://doi.org/10.1162/nol_a_00137)." In _Neurobiology of Language_ 5(1), pp. 43–63, 2024.
- Clark, Oh, and Schuler. "[Linear Recency Bias During Training Improves Transformers' Fit to Reading Times](https://aclanthology.org/2025.coling-main.517/)." In _Proceedings of the 31st International Conference on Computational Linguistics (COLING)_, 2025.
- Oh, Zhu, and Schuler. "[The Inverse Scaling Effect of Pre-Trained Language Model Surprisal Is Not Due to Data Leakage](https://aclanthology.org/2025.findings-acl.91/)." In _Findings of the Association for Computational Linguistics: ACL 2025_, 2025.
- Wang, Jiang, Linzen, and Lake. "[Rapid Word Learning Through Meta In-Context Learning](https://aclanthology.org/2025.emnlp-main.1631/)." In _Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing_, 2025.

### Domain-Specific Models, Benchmarks, and Applications

#### Formal Systems

- Kolak, Martins, Le Goues, and Hellendoorn. "[Patch Generation with Language Models: Feasibility and Scaling Behavior](https://openreview.net/forum?id=rHlzJh_b1-5)." In _Deep Learning for Code Workshop @ ICLR_, 2022.
- Xu, Alon, Neubig, and Hellendoorn. "[A Systematic Evaluation of Large Language Models of Code](https://doi.org/10.1145/3520312.3534862)." In _Proceedings of the 6th ACM SIGPLAN International Symposium on Machine Programming_, 2022.
- Feng, Luo, Wang, Tang, Yang, Shao, Mguni, Du, and Wang. "[ChessGPT: Bridging Policy Learning and Language Modeling](https://proceedings.neurips.cc/paper_files/paper/2023/hash/16b14e3f288f076e0ca73bdad6405f77-Abstract.html)." In _Advances in Neural Information Processing Systems 36: Datasets and Benchmarks Track_, 2023.
- Rao, Jain, Alon, Le Goues, and Hellendoorn. "[CAT-LM: Training Language Models on Aligned Code And Tests](https://doi.org/10.1109/ASE56229.2023.00193)." In _38th IEEE/ACM International Conference on Automated Software Engineering_, pp. 409–420, 2023.
- Di, Li, Yu, Jiang, Cai, Cao, Chen, Chen, Chen, Chen, Fan, Gong, Gong, Hu, Guo, Lei, Li, Li, Liang, Liao, Liu, Liu, Liu, Lu, Shen, Wang, Wang, Wang, Xu, Yang, Ye, Zhang, Zhang, Zhao, Zheng, Zhou, Zhu, and Zhu. "[CodeFuse-13B: A Pretrained Multi-lingual Code Large Language Model](https://doi.org/10.1145/3639477.3639719)." In _46th International Conference on Software Engineering: Software Engineering in Practice_, pp. 418–429, 2024.
- Kadosh, Hasabnis, Vo, Schneider, Krien, Capota, Wasay, Tamir, Willke, Ahmed, Pinter, Mattson, and Oren. "[MonoCoder: Domain-Specific Code Language Model for HPC Codes and Tasks](https://doi.org/10.1109/HPEC62836.2024.10938441)." In _IEEE High Performance Extreme Computing Conference_, pp. 1–7, 2024.
- Paul, Yang, Glavaš, Kersting, and Gurevych. "[ObscuraCoder: Powering Efficient Code LM Pre-Training Via Obfuscation Grounding](https://proceedings.iclr.cc/paper_files/paper/2025/hash/23ff02034404b65776080cbf7148addd-Abstract-Conference.html)." In _International Conference on Learning Representations_, 2025.

#### Multilingual and Language-Specific Models

- **Ko**, **Yang**, **Ryu**, **Choi**, **Yang**, and Park. "[A Technical Report for Polyglot-Ko: Open-Source Large-Scale Korean Language Models](https://arxiv.org/abs/2306.02254)." _arXiv preprint arXiv:2306.02254_, 2023.
- Yang, Dodé, Ferenczi, Héja, Jelencsik-Mátyus, Kőrös, Laki, Ligeti-Nagy, Vadász, and Váradi. "[Jönnek a nagyok! BERT-Large, GPT-2 és GPT-3 nyelvmodellek magyar nyelvre](https://acta.bibl.u-szeged.hu/78417/)." In _XIX. Magyar Számítógépes Nyelvészeti Konferencia (MSZNY 2023)_, pp. 247-262, 2023.
- Caillaut, Nakhlé, Qader, Liu, and Barthélemy. "[Scaling Laws of Decoder-Only Models on the Multilingual Machine Translation Task](https://aclanthology.org/2024.wmt-1.124/)." In _Proceedings of the Ninth Conference on Machine Translation_, 2024.
- Etxaniz, Sainz, Pérez, Aldabe, Rigau, Agirre, Ormazabal, Artetxe, and Soroa. "[Latxa: An Open Language Model and Evaluation Suite for Basque](https://aclanthology.org/2024.acl-long.799/)." In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, 2024.
- Ji, Li, Paavola, Lin, Chen, O'Brien, Luo, Schütze, Tiedemann, and Haddow. "[EMMA-500: Enhancing Massively Multilingual Adaptation of Large Language Models](https://arxiv.org/abs/2409.17892)." In _arXiv preprint arXiv:2409.17892_, 2024.
- Bergmanis, Pretkalniņš, Kronis, Nicmanis, Jelinska, Rozis, Vīksna, and Pinnis. "[TildeOpen LLM: Leveraging Curriculum Learning to Achieve Equitable Language Representation](https://aclanthology.org/2026.lrec-1.834/)." In _Proceedings of the Fifteenth Language Resources and Evaluation Conference_, 2026.

#### AI4Science

- Azerbayev, Piotrowski, and Avigad. "[ProofNet: A Benchmark for Autoformalizing and Formally Proving Undergraduate-Level Mathematics Problems](https://neurips.cc/virtual/2022/58291)." In _Math-AI Workshop @ NeurIPS_, 2022.
- Horawalavithana, Ayton, Sharma, Howland, Subramanian, Vasquez, Cosbey, Glenski, and Volkova. "[Foundation Models of Scientific Knowledge for Chemistry: Opportunities, Challenges and Lessons Learned](https://aclanthology.org/2022.bigscience-1.12/)." In _Proceedings of BigScience Episode #5 — Workshop on Challenges & Perspectives in Creating Large Language Models_, 2022.
- Dollar, Horawalavithana, Vasquez, Pfaendtner, and Volkova. "[MolJET: Multimodal Joint Embedding Transformer for Conditional de novo Molecular Design and Multi-Property Optimization](https://openreview.net/forum?id=7UudBVsIrr)." _ICLR 2023 submission/preprint_, 2023.
- **Azerbayev**, **Schoelkopf**, Paster, Dos Santos, McAleer, Jiang, Deng, **Biderman**, and Welleck. "[Llemma: An Open Language Model for Mathematics](https://proceedings.iclr.cc/paper_files/paper/2024/hash/b225f5c7cd13615e9558c3931fa4e66f-Abstract-Conference.html)." In _International Conference on Learning Representations_, 2024.
- de Haan, Ting, Ghosal, Nguyen, Accomazzi, Herron, Lama, Pan, Wells, and Ramachandra. "[AstroSage: Leading Performance in Astronomy Q&A with a 70B-Parameter Domain-Specialized Model](https://ml4astro.github.io/icml2025/assets/camera_ready/12_AstroSage_Leading_Performan.pdf)." In _Workshop on Machine Learning for Astrophysics @ ICML_, 2025.
- Brixi et al. "[Genome Modelling and Design across All Domains of Life with Evo 2](https://www.nature.com/articles/s41586-026-10176-5)." _Nature_ 652, pp. 1349–1361, 2026. Evo 2 used Savanna, a training stack built in part from GPT-NeoX components.
- Zamora-Resendiz, Yin, Kimbrel, Beckham, Million Veteran Program, and Crivelli. "[A Longitudinal Clinical Foundation Model on Nationwide Veteran Health Trajectories](https://www.medrxiv.org/content/10.64898/2026.05.13.26353133v1)." _medRxiv preprint 2026.05.13.26353133_, 2026.

## Models

The following models were trained with GPT-NeoX. Entries that used a framework derived from GPT-NeoX say so.

### English LLMs

- EleutherAI's [GPT-NeoX-20B](https://huggingface.co/EleutherAI/gpt-neox-20b) (20B; trained on 96x A100 40GB GPUs on CoreWeave) and [Pythia (70M through 12B)](https://github.com/EleutherAI/pythia) (6.9B and 12B models trained on A100 40GB GPUs)
- CarperAI's [FIM-NeoX-1.3B](https://huggingface.co/CarperAI/FIM-NeoX-1.3B)
- StabilityAI's [StableLM (3B and 7B)](https://github.com/Stability-AI/StableLM) (7B model trained on an undisclosed Stability AI cluster) and [StableLM 2 (12B)](https://huggingface.co/stabilityai/stablelm-2-12b) (trained on 384x H100 GPUs on AWS P5)
- Together.ai's [RedPajama-INCITE (3B and 7B)](https://together.ai/blog/redpajama-models-v1) (7B model trained on 3,072x V100 GPUs on Summit at OLCF)
- Together AI's [StripedHyena-Hessian-7B](https://huggingface.co/togethercomputer/StripedHyena-Hessian-7B) (English; trained on 256 GPUs with Savanna, a separate framework built in part from GPT-NeoX components; public checkpoint)
- Carnegie Mellon University's [proofGPT (1.3B and 6.7B)](https://huggingface.co/hoskinson-center/proofGPT-v0.1-6.7B) (training cluster undisclosed)
- [Hubble (8B)](https://arxiv.org/abs/2510.19811) (trained on an A100 80GB cluster)
- Geodesic Research's [Alignment Pretraining (6.9B)](https://alignmentpretraining.ai/) (pretrained on 256x GH200 GPUs; SFT on 64x H100 GPUs)
- EleutherAI's [Deep Ignorance (6.9B)](https://huggingface.co/collections/EleutherAI/deep-ignorance) (filtered and unfiltered models trained from scratch on 128x H100 GPUs for 500B pretraining tokens plus 50B annealing tokens; public checkpoints)
- Dampish's [StellarX (2.8B and 4B)](https://huggingface.co/Dampish/StellarX-4B-V0.2)
- Chinese Academy of Sciences's [AstroSNN (1.5B)](https://arxiv.org/abs/2312.07625)
- Chinese Academy of Sciences' [Quest (1.4B, 6.9B, and 12B)](https://arxiv.org/abs/2405.19846) long-context models (continued training of Pythia at 32K and 128K context; no public checkpoints located)
- Chinese Academy of Sciences' [NExtLong (8B)](https://huggingface.co/collections/caskcsg/nextlong-models) (continued pre-training of Llama 3 at up to 512K context; 128K and 512K variants trained on H100 GPUs)
- camgeodesic's [OLMo-3 7B Sycophancy Inoculation models](https://huggingface.co/camgeodesic/olmo3_7b_sycophancy_replay_only_control) (three continued-pretraining runs on 256x GH200 GPUs; public checkpoints)
- HPLT's [HPLT v2 corpus-comparison ablation models (1.82B)](https://hplt-project.org/hplt-v2-dataset-performance) (historical GPT-NeoX runs on 32x AMD MI250X GPUs on LUMI; distinct from the later 1.7B reference models trained with Megatron-based stacks)

### Non-English LLMs

- EleutherAI's [Polyglot-Ko (1.3B through 12.8B)](https://github.com/EleutherAI/polyglot) (Korean; 12.8B model trained on 256x A100 GPUs on a Stability AI cluster)
- Korea University's [KULLM-Polyglot (5.8B and 12.8B)](https://github.com/nlpai-lab/KULLM) (Korean; training cluster undisclosed)
- Stability AI's [Japanese Stable LM (7B)](https://huggingface.co/stabilityai/japanese-stablelm-base-alpha-7b) (Japanese; training cluster undisclosed, Stability AI cluster)
- LearnItAnyway's [LLaVA-Polyglot-Ko (1.3B)](https://huggingface.co/LearnItAnyway/llava-polyglot-ko-1.3b-hf) (Korean)
- Rinna Co.'s [japanese-gpt-neox-3.6b](https://huggingface.co/rinna/japanese-gpt-neox-3.6b) (Japanese) and [bilingual-gpt-neox-4b](https://huggingface.co/rinna/bilingual-gpt-neox-4b) (English / Japanese)
- Rinna Co.'s [youri-7b](https://huggingface.co/rinna/youri-7b) (Japanese / English; continual pre-training of Llama 2 7B on approximately 40B tokens using code based on GPT-NeoX; public checkpoint)
- Rinna Co.'s [japanese-gpt-neox-small (0.2B)](https://huggingface.co/rinna/japanese-gpt-neox-small) (Japanese; trained from scratch using code based on GPT-NeoX; public checkpoint)
- CyberAgent's [OpenCALM (125M through 7B)](https://huggingface.co/cyberagent/open-calm-7b) (Japanese; training cluster undisclosed)
- The Hungarian Research Centre for Linguistics's [PULI GPTrio (6.7B)](https://huggingface.co/NYTK/PULI-GPTrio) (Hungarian / English / Chinese; training cluster undisclosed)
- The University of Tokyo Matsuo Lab's [weblab-10b](https://huggingface.co/matsuo-lab/weblab-10b) ([earlier checkpoint namespace](https://huggingface.co/Kojima777/weblab-10b)) and [weblab-10b-instruct](https://huggingface.co/Kojima777/weblab-10b-instruction-sft) (Japanese; training cluster undisclosed)
- nolano.ai's [Hi-NOLIN (9B)](https://blog.nolano.ai/Hi-NOLIN/) (English / Hindi; trained on 3,072 GPUs across 512 nodes on Summit at OLCF under an INCITE allocation)
- Renmin University of China's [YuLan (12B)](https://huggingface.co/yulan-team/YuLan-Base-12b) (English, Chinese; training cluster undisclosed)
- The Basque Center for Language Technology's [Latxa (7B through 70B)](https://huggingface.co/HiTZ/latxa-70b-v1.2) (Basque; trained on A100 64GB GPUs on Leonardo at CINECA / EuroHPC)
- MaLA-LM's [EMMA-500 (7B and 8B)](https://github.com/MaLA-LM/emma-500) (multilingual, 546 languages; trained on 256x AMD MI250X GPUs on LUMI)
- Tilde AI's [TildeOpen (30B)](https://huggingface.co/TildeAI/TildeOpen-30b) (European languages; trained on 768x AMD MI250X GPUs on LUMI, approximately 1.5M GPU-hours)
- SB Intuitions' [Sarashina1 (7B, 13B, and 65B)](https://huggingface.co/sbintuitions/sarashina1-65b) (Japanese; 13B trained on 1T tokens and 65B on 800B tokens; public MIT-licensed checkpoints)
- Rinna Co.'s [Llama 3 Youko 8B](https://huggingface.co/rinna/llama-3-youko-8b) and [70B](https://huggingface.co/rinna/llama-3-youko-70b) (Japanese / English; continual pre-training of Llama 3 on approximately 5B tokens; public checkpoints)
- Rinna Co.'s [Nekomata-7B](https://huggingface.co/rinna/nekomata-7b) (Japanese / English; continual pre-training of Qwen-7B on approximately 30B tokens; public checkpoint)
- ABEJA's [GPT-NeoX-Japanese (2.7B)](https://huggingface.co/abeja/gpt-neox-japanese-2.7b) (Japanese; its official card and training blog indicate use of the GPT-NeoX codebase; public checkpoint)
- Bit192's [Japanese 20B model](https://www.coreweave.com/resources/case-studies/coreweave-and-bit192-help-gpt-neox-20b-reach-japan) (Japanese; trained from scratch with a slightly modified GPT-NeoX codebase on NVIDIA A100 and A40 GPUs on CoreWeave and deployed through AI Novelist; no public checkpoint)

### Code Models

- Carnegie Mellon University's [PolyCoder (160M through 2.7B)](https://github.com/VHellendoorn/Code-LMs) and [CAT-LM (2.7B)](https://huggingface.co/nikitharao/catlm)
- StabilityAI's [StableCode (1.3B)](https://stability.ai/blog/stablecode-llm-generative-ai-coding) and [StableCode-Completion-Alpha (3B)](https://stability.ai/blog/stablecode-llm-generative-ai-coding)
- CodeFuse AI's [CodeFuse (13B)](https://huggingface.co/codefuse-ai/CodeFuse-13B) (training cluster undisclosed)
- DFKI and TU Darmstadt's [ObscuraCoder (255M through 2.8B)](https://arxiv.org/abs/2504.00019) (trained from scratch on 272B tokens; no public checkpoints located)

### AI for Science

- EleutherAI's [LLeMMA (7B and 34B)](https://arxiv.org/abs/2310.10631) (trained on 256x A100 40GB GPUs)
- Oak Ridge National Lab's [FORGE](https://github.com/at-aaims/forge) models for science. The suite has 13 models with 1.44B, 13B, and 25.6B parameters, including six 1.44B domain-specific models. They were trained on 257B tokens from scientific articles using GPT-NeoX on up to 2,048 AMD MI250X GCDs on Frontier at OLCF. The [SC '23 paper](https://doi.org/10.1145/3581784.3613215) reports models up to 26B and the repository lists models up to 22.4B. The paper also introduced three downstream tasks for science, which are domain classification, material phase classification, and energy regression. See also the [ORNL publication page](https://www.ornl.gov/publication/forge-pre-training-open-foundation-models-science).
- Oak Ridge National Lab's [Unnamed Material Science Domain Models (7B)](https://arxiv.org/abs/2402.00691) (trained on AMD MI250X GPUs on Frontier at OLCF)
- AstroMLab's [AstroSage-Llama-3.1-70B](https://arxiv.org/abs/2505.17592) (70B; astronomy; trained on AMD MI250X GPUs on Frontier at OLCF)
- Pacific Northwest National Lab's [MolJET (undisclosed size)](https://openreview.net/pdf?id=7UudBVsIrr)
- The Arc Institute and Together AI's [Evo 1 (7B)](https://huggingface.co/togethercomputer/evo-1-8k-base) (genomics; trained on approximately 300B tokens and 256 GPUs with Savanna, a separate framework built in part from GPT-NeoX components; public 8K- and 131K-context checkpoints)
- The Arc Institute-led collaboration's [Evo 2 (7B and 40B)](https://huggingface.co/arcinstitute/savanna_evo2_40b) (genomics; the 40B model was trained on 9.3T tokens and 2,048 GPUs using Savanna, a training stack built in part from GPT-NeoX components)
- Lawrence Berkeley National Lab, Oak Ridge National Lab, and the U.S. Department of Veterans Affairs' [VA-LLM (1.62B)](https://www.medrxiv.org/content/10.64898/2026.05.13.26353133v1) (clinical text; trained from scratch on 1.74T tokens using a modified GPT-NeoX framework; no public checkpoint)

### Other Modalities

- Rinna Co.'s [PSLM (7B)](https://arxiv.org/abs/2406.12428) (speech / text; training cluster undisclosed)
- University College London's [ChessGPT-3B](https://huggingface.co/Waterhorse/chessgpt-base-v1)
- Gretel's [Text-to-Table (3B)](https://huggingface.co/gretelai/text2table)

## NeoX-Derived Frameworks and Training Extensions

The following projects are forks or extensions of GPT-NeoX.

- [Savanna](https://github.com/Zymrael/savanna), an alternative-architecture pretraining framework built from GPT-NeoX's general repository structure and configuration alongside DeepSpeed and Transformer Engine; used for StripedHyena, Evo 1, and Evo 2.
- IBM's [Times-NeoX](https://github.com/IBM/times-neox), a GPT-NeoX fork for univariate time-series forecasting that adapts NeoX's training and generation scripts; no named released checkpoint or parameter count was identified.
- Chandar Lab's [continual-pretraining pipeline](https://github.com/chandar-lab/continual-pretraining), a GPT-NeoX / Megatron-DeepSpeed extension for continual pre-training with replay and gradient alignment.
- Tilde's [llm-gpt-neox](https://github.com/tilde-nlp/llm-gpt-neox), a maintained GPT-NeoX branch used to train and fine-tune the TildeOpen 30B model line on LUMI.
- Oak Ridge National Lab ported GPT-NeoX to Frontier to train the FORGE models, as described in the [SC '23 paper](https://doi.org/10.1145/3581784.3613215). The port added a Slurm launcher with affinity binding and support for the LAMB optimizer. It also converted the masking and softmax fusion kernels to HIP for AMD GPUs and moved to the mainstream version of DeepSpeed.

## Adding Your Work

To add a model or publication, open a pull request that edits this file. Follow the format of the existing entries and link to a paper, model card, or repository that states GPT-NeoX was used. If you used a fork of GPT-NeoX, mention it in the entry.

```
- <Organization>'s [<Model> (<sizes>)](<url>) (<language or domain>; trained on <hardware>)
```

If an entry is incorrect, please open an issue or pull request.
