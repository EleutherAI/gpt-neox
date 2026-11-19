# RAG Retrieval & Context Pipeline (retrievers / reranking / context compression / long-context vs RAG / efficiency) — as of 2026-11-19

> **Method note for the report writer (read first).** In this session, direct fetches of `arxiv.org` and `export.arxiv.org` were blocked (DNS not resolvable in the session's allowed network), so no abs page, OpenReview forum, or ACL Anthology page could be opened directly. All verification below comes from web-search results that index **primary pages** (iclr.cc / neurips.cc / icml.cc virtual poster pages, proceedings.mlr.press, ACL Anthology preview pages, arXiv comments/PDF headers as rendered by the search index). Quoted abstract phrases were cross-checked against the abstract text that the search index returned. Where the exact day of first arXiv release could not be confirmed, only the month (from the arXiv ID) is given. Venue labels are given only when a primary page confirmed them; otherwise "Preprint (arXiv)" or "Unverified".

## Q1. What are the most influential and most recent (2025–2026) papers improving the retrieve → rerank → compress → generate pipeline, and what venue accepted each?

### Takeaway
A 12-paper list covers the pipeline end to end. Two foundational early-2025 papers (ReasonIR and Rank1) started the "reasoning-intensive retrieval and reranking" line. The 2025–2026 papers that followed were accepted at ICLR 2026 (LIMIT; *Rethinking Reasoning in Document Ranking*), ACL 2026 Main (ReasonEmbed, ReasonRank), WWW 2026 (SeleCom), ICML 2026 (ProphetKV) and NeurIPS 2026 (OpticalRAG). REFRAG (Meta), CLaRa (Apple) and a 2026 long-context-vs-RAG cost study are still preprints, or have only a workshop acceptance.

### Cited Findings

#### Summary table

| # | Paper (short) | First author et al. | arXiv | First release | Verified venue status | Sub-category |
|---|---|---|---|---|---|---|
| A1 | ReasonIR **[Foundational]** | Rulin Shao et al. | 2504.20595 | 2025-04-29 | **Unverified** (one secondary source says COLM 2025) | A. Retrieval quality |
| A2 | On the Theoretical Limitations of Embedding-Based Retrieval (LIMIT) | Orion Weller et al. | 2508.21038 | 2025-08 | **ICLR 2026 (Poster)** | A. Retrieval quality |
| A3 | ReasonEmbed | Jianlyu Chen et al. | 2510.08252 | 2025-10 | **ACL 2026 Main (Long)** | A. Retrieval quality |
| B1 | Rank1 **[Foundational]** | Orion Weller et al. | 2502.18418 | 2025-02-25 | **COLM 2025** | B. Reranking |
| B2 | ReasonRank | Wenhan Liu et al. | 2508.07050 | 2025-08-09 | **ACL 2026 Main (Long)** | B. Reranking |
| B3 | Rethinking Reasoning in Document Ranking: Why CoT Falls Short | Xuan Lu et al. | 2510.08985 | 2025-10 | **ICLR 2026 (Poster)** (ICLR page title: "The Overthinking Predicament: When Reasoning Hurts Ranking") | B. Reranking |
| C1 | SeleCom (Rethinking Soft Compression in RAG) | Yunhao Liu et al. | 2602.15856 | 2026-01-25 | **WWW 2026** | C. Context compression |
| C2 | CLaRa | Jie He et al. | 2511.18659 | 2025-11-24 | **Preprint (arXiv)** | C. Compression + joint retriever–generator training |
| C3 | OpticalRAG | Senhao Liu et al. | 2611.13675 | 2026-11 | **NeurIPS 2026 (Poster)** | C. Compression (pixel-space) |
| D1 | The Token Tax of Epistemic Accuracy | Austin Hamilton et al. | 2606.20898 | 2026-06-18 | **Preprint (arXiv)** | D. Long-context vs RAG |
| E1 | REFRAG | Xiaoqiang Lin et al. | 2509.01092 | 2025-09-01 | **NeurIPS 2025 Workshop (MTI-LLM) Poster**; main-track venue **Unverified** | E. Efficiency (decoding) |
| E2 | ProphetKV | Shihao Wang et al. | 2602.02579 | 2026-01-31 | **ICML 2026 (Poster)**, PMLR v306 | E. Efficiency (KV-cache reuse) |

---

#### A. Retrieval quality (retrievers / embeddings)

**[A1] ReasonIR: Training Retrievers for Reasoning Tasks** — *Foundational (early 2025)*
- Authors: Rulin Shao et al. (Meta). arXiv: [2504.20595](https://arxiv.org/abs/2504.20595). First release: 2025-04-29 (Meta's publication page is dated 2025-04-25) — [Meta AI](https://ai.meta.com/research/publications/reasonir-training-retrievers-for-reasoning-tasks/); [HF Papers](https://huggingface.co/papers/2504.20595)
- Venue: **Unverified.** Neither the arXiv record nor Meta's page mentions a venue in the search results. One secondary source, an IR reading-group slide deck, lists it as "COLM 2025". The COLM proceedings and OpenReview were not confirmed — [Speaker Deck (secondary)](https://speakerdeck.com/ynakano/irreadings2025fall-information-retrieval-and-reasoning-tasks)
- Korean summary:
  1. 추론이 필요한 질의에서 기존 검색기가 약한 문제를 다룬다. 문서마다 "추론해야 매칭되는 어려운 질의"와 "그럴듯하지만 실제로는 쓸모없는 hard negative"를 합성하는 ReasonIR-Synthesizer를 만들고, 이 데이터와 공개 데이터로 Llama3.1-8B 기반 bi-encoder ReasonIR-8B를 학습했다.
  2. BRIGHT에서 리랭커 없이 nDCG@10 29.9, 리랭커를 쓰면 36.9를 기록했다. RAG에 적용하면 closed-book보다 MMLU가 6.4%, GPQA가 22.6% 올랐다.
- Quote: "a challenging and relevant query that requires reasoning to match, as well as a plausibly related but ultimately unhelpful hard negative" — [HF Papers](https://huggingface.co/papers/2504.20595)
- Caveat: a third-party summary (alphaXiv) says the basic setting scores 24.4 nDCG@10 and the 29.9 figure uses query rewriting — [search-indexed alphaXiv overview](https://alphaxiv.org/overview/2504.20595v1)

**[A2] On the Theoretical Limitations of Embedding-Based Retrieval** (LIMIT dataset)
- Authors: Orion Weller, Michael Boratko, Iftekhar Naim, Jinhyuk Lee (Google DeepMind / JHU). arXiv: [2508.21038](https://arxiv.org/abs/2508.21038). First release: 2025-08 (the exact day was not confirmed)
- Venue: **ICLR 2026 (Poster)** — [ICLR 2026 virtual poster](https://iclr.cc/virtual/2026/poster/10007818); [OpenReview PDF](https://openreview.net/pdf/43db7db7a837ff1e0af8065943aec2d036382459.pdf); [mlanthology ICLR 2026 entry](https://mlanthology.org/iclr/2026/weller2026iclr-theoretical/)
- Korean summary:
  1. 단일 벡터 임베딩 검색의 이론적 한계를 학습이론 결과와 연결해 분석한다. 어떤 질의로 반환될 수 있는 top-k 문서 부분집합의 수가 임베딩 차원에 의해 제한된다는 것을 보였다. 테스트셋에 직접 최적화한 free embedding에서도 이 한계가 그대로 나타났다.
  2. 이 이론을 토대로 단순한 과제로 구성한 LIMIT 데이터셋을 만들었는데, 최신(SOTA) 임베딩 모델들이 이 데이터셋에서 실패했다. 지시 기반의 일반 질의를 다루려면 단일 벡터 패러다임을 넘어서는 방법(cross-encoder, multi-vector 등)이 필요하다고 제안한다.
- Quote: "even state-of-the-art models fail on this dataset despite the simple nature of the task" — [HF Papers](https://huggingface.co/papers/2508.21038)
- Additional (summary-level, not abstract): SoTA MTEB models score below 20 recall@100 on LIMIT; code and data are at github.com/google-deepmind/limit — [OpenReview PDF](https://openreview.net/pdf/43db7db7a837ff1e0af8065943aec2d036382459.pdf)

**[A3] ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval**
- Authors: Jianlyu Chen, Junwei Lan, Chaofan Li, Defu Lian, Zheng Liu. arXiv: [2510.08252](https://arxiv.org/abs/2510.08252). First release: 2025-10
- Venue: **ACL 2026 Main (Long)**, Anthology ID 2026.acl-long.54, pp. 1203–1221 — [ACL Anthology (preview)](https://preview.aclanthology.org/ingest-acl/2026.acl-long.54/); [ACL Anthology PDF](https://aclanthology.org/2026.acl-long.54.pdf)
- Korean summary:
  1. 기존 합성 학습 데이터는 너무 쉬운 예제가 많다는 '자명성(triviality)' 문제가 있다. 이를 해결하는 데이터 합성 기법 ReMixer로 고품질 샘플 82K개를 만들고, 샘플마다 추론 강도에 따라 가중치를 조절하는 Redapter로 추론 집약 검색용 임베딩 모델을 학습했다.
  2. ReasonEmbed-Qwen3-8B는 BRIGHT에서 nDCG@10 38.1로 기존 텍스트 임베딩 모델을 크게 앞섰다(R2MED에서는 43.18). 모델과 데이터 같은 자원은 모두 공개한다고 밝혔다.
- Quote: "a record-high nDCG@10 score of 38.1 on the BRIGHT benchmark, which significantly outperforms existing text embedding models" — [arXiv HTML](https://arxiv.org/html/2510.08252v2)

#### B. Reranking

**[B1] Rank1: Test-Time Compute for Reranking in Information Retrieval** — *Foundational (early 2025)*
- Authors: Orion Weller, Kathryn Ricci, Eugene Yang, Andrew Yates, Dawn Lawrie, Benjamin Van Durme (JHU). arXiv: [2502.18418](https://arxiv.org/abs/2502.18418v1). First release: 2025-02-25 (revised 2025-08-08)
- Venue: **COLM 2025.** The paper's camera-ready header reads "Published as a conference paper at COLM 2025" — [alphaXiv rendering of arXiv v2](https://www.alphaxiv.org/abs/2502.18418.md)
- Korean summary:
  1. 추론 LM(DeepSeek-R1)이 MS MARCO 질의·패시지에 대해 생성한 추론 흔적 60만 개 이상을 작은 모델에 증류했다. 이렇게 test-time compute를 활용하도록 학습한 최초의 리랭커 Rank1을 제안한다.
  2. 추론 체인을 사용자나 RAG 시스템에 그대로 보여줄 수 있어 설명이 가능하고, 양자화 버전도 적은 연산·메모리로 강한 성능을 유지했다. 또 기존 relevance 판정 데이터셋이 이미 포화되었을 가능성을 지적했다.
- Quote: "the first reranking model trained to take advantage of test-time compute" — [arXiv](https://arxiv.org/abs/2502.18418v1)

**[B2] ReasonRank: Empowering Passage Ranking with Strong Reasoning Ability**
- Authors: Wenhan Liu, Xinyu Ma, Weiwei Sun, Yutao Zhu, Yuchen Li, Dawei Yin, Zhicheng Dou. arXiv: [2508.07050](https://arxiv.org/abs/2508.07050v1). First release: 2025-08-09 (v2 2025-08-22)
- Venue: **ACL 2026 Main (Long)**, Anthology ID 2026.acl-long.1005 — [ACL Anthology (preview)](https://preview.aclanthology.org/ingest-acl/2026.acl-long.1005/); [ACL Anthology PDF](https://aclanthology.org/2026.acl-long.1005.pdf)
- Korean summary:
  1. 추론 집약 랭킹용 학습 데이터가 부족한 문제를 다룬다. DeepSeek-R1로 positive·hard negative 패시지를 찾아 학습용 패시지 리스트를 만들고, 추론 체인과 정답 랭킹을 레이블로 생성했다. 이 데이터로 cold-start SFT 다음에 RL을 거치는 2단계 학습을 해 listwise 추론 리랭커를 만들었다.
  2. BRIGHT와 R2MED에서 기존 베이스라인을 크게 앞섰고, pointwise 리랭커 Rank1보다 지연이 훨씬 낮았다. BRIGHT 리더보드에서는 40.6으로 SOTA를 기록했다.
- Quote: "ReasonRank outperforms existing baselines significantly and also achieves much lower latency than pointwise reranker Rank1" — [arXiv HTML](https://arxiv.org/html/2508.07050v1)

**[B3] Rethinking Reasoning in Document Ranking: Why Chain-of-Thought Falls Short** (ICLR poster title: *The Overthinking Predicament: When Reasoning Hurts Ranking*)
- Authors: Xuan Lu, Haohang Huang, Rui Meng, Yaohui Jin, Wenjun Zeng, Xiaoyu Shen (the author list comes from the mlanthology BibTeX). arXiv: [2510.08985](https://arxiv.org/html/2510.08985v1). First release: 2025-10
- Venue: **ICLR 2026 (Poster)** — [ICLR 2026 virtual poster](https://iclr.cc/virtual/2026/poster/10006933); [mlanthology ICLR 2026](https://mlanthology.org/iclr/2026/lu2026iclr-rethinking/)
- Korean summary:
  1. CoT 추론을 넣은 리랭커가 실제로 도움이 되는지 pointwise와 listwise 양쪽에서 같은 조건으로 체계적으로 비교 분석했다.
  2. 추론을 넣은 리랭커가 CoT 없이 바로 랭킹하는 모델보다 일관되게 성능이 낮았다. 원인으로 pointwise에서는 점수 보정이 무너져 양성 쪽으로 치우치는 문제(TPR은 오르고 TNR은 떨어짐)를, listwise에서는 과적합과 과사고(overthinking)를 꼽았다. 대안으로 보정을 고려한 점수화와 간결하고 목표 지향적인 추론을 제안한다.
- Quote: "reasoning-augmented rerankers consistently underperform their direct counterparts that predict rankings without CoT" — [arXiv HTML](https://arxiv.org/html/2510.08985v1)

#### C. Context compression / pruning and joint retriever–generator training

**[C1] Rethinking Soft Compression in Retrieval-Augmented Generation: A Query-Conditioned Selector Perspective** (SeleCom)
- Authors: Yunhao Liu, Zian Jia, Xinyu Gao, Kanjun Xu, Yun Xiong. arXiv: [2602.15856](https://arxiv.org/abs/2602.15856). First release: 2026-01-25
- Venue: **WWW 2026.** The arXiv comments say it was accepted by WWW 2026. The PDF header gives WWW '26, Dubai, Apr 13–17 2026, DOI 10.1145/3774904.3792700 — [arXiv](https://arxiv.org/abs/2602.15856); [arXiv PDF](https://www.arxiv.org/pdf/2602.15856)
- Korean summary:
  1. 기존 soft compression은 질의와 상관없이 문서 전체를 임베딩으로 압축해서 비압축 RAG보다 성능이 떨어진다. 저자들은 그 원인을 두 가지로 본다. 전체 압축은 LLM의 생성 방식과 충돌하고, 질의와 관련된 정보만 있으면 되므로 애초에 불필요하다. 그래서 인코더를 질의 조건부 selector(디코더 전용)로 바꾼 SeleCom을 제안하고, 난이도별로 나눈 대규모 합성 QA 데이터로 커리큘럼 학습을 했다.
  2. 기존 soft compression 방법들을 능가하고 비압축 베이스라인과 같거나 더 나은 성능을 내면서, 연산량과 지연을 33.8~84.6% 줄였다.
- Quote: "full-compression conflicts with the LLM's downstream generation behavior" (wording as returned by the search index's rendering of the abstract) — [arXiv HTML](https://arxiv.org/html/2602.15856v1)

**[C2] CLaRa: Bridging Retrieval and Generation with Continuous Latent Reasoning**
- Authors: Jie He, Richard He Bai, Sinead Williamson, Jeff Z. Pan, Navdeep Jaitly, Yizhe Zhang (Apple / Univ. of Edinburgh). arXiv: [2511.18659](https://arxiv.org/abs/2511.18659). First release: 2025-11-24 (v2 11-25; v3 exists)
- Venue: **Preprint (arXiv).** No journal-ref or acceptance was found — [arXiv](https://arxiv.org/abs/2511.18659)
- Korean summary:
  1. 검색기와 생성기가 따로 최적화되는 문제를 다룬다. 문서를 한 번만 압축해 만든 연속 메모리 토큰을 검색과 생성에 함께 쓴다. 연속 표현이라 Straight-Through 추정으로 top-k 선택을 미분 가능하게 만들 수 있고, 덕분에 하나의 next-token 손실로 질의 인코더(리랭커)와 생성기를 end-to-end로 같이 학습한다.
  2. 여러 QA 벤치마크에서 텍스트 압축률 16에서도 압축과 리랭킹 모두 SOTA 성능을 냈고, 텍스트 기반으로 미세조정한 베이스라인을 능가했다. 코드는 apple/ml-clara에 공개했다.
- Quote: "embedding-based compression and joint optimization in a shared continuous space" — [arXiv](https://arxiv.org/abs/2511.18659)
- Caveat: secondary summaries quote other compression ratios (32–64×, up to 256×). Use the paper's own numbers — [hyper.ai](https://hyper.ai/papers/2511.18659)

**[C3] OpticalRAG: Pixel-Space Compression for Token-Efficient Retrieval-Augmented Generation**
- Authors: Senhao Liu, Yuheng Zhang, Chunyu Wei, Yueguo Chen, Xinran Zhang. arXiv: [2611.13675](https://arxiv.org/abs/2611.13675v2). First release: 2026-11 (v2 already posted). The arXiv ID was seen only as a search-index link title; the abs page itself was not opened.
- Venue: **NeurIPS 2026 (Poster)** — [NeurIPS 2026 virtual poster](https://neurips.cc/virtual/2026/poster/152209)
- Korean summary:
  1. 문서를 페이지 이미지로 렌더링해 고정된(frozen) 비전 인코더로 인코딩한다. 텍스트 질의로 시각 청크를 검색하는 Encoder-Bridged Retrieval(대조 학습 없이 VLM의 공유 공간 활용)과 질의 조건부 cross-attention 압축기인 Query-Driven Distillation을 써서, 검색된 시각 토큰을 16개까지 줄인 뒤 LLM에 넣는다.
  2. LongBench-v2, BAMBOO, LooGLE-v2에서 16개 토큰만 넣고도 압축 대비 정확도(CAP)가 가장 높았다. closed-book Qwen2.5-7B-Instruct보다 LongBench-v2에서 5.17점 높았다.
- Quote: "distills the retrieved visual tokens into as few as 16 injected tokens" — [NeurIPS 2026 poster page](https://neurips.cc/virtual/2026/poster/152209)
- Scope note: this compresses text corpora by rendering them as images. It is a context-compression paper, but it borders on multimodal RAG.

#### D. Long-context LLMs vs RAG

**[D1] The Token Tax of Epistemic Accuracy: Comparing RAG and Long-Context Architectures for Document-Grounded Generative AI Applications**
- Authors: Austin Hamilton, Ryan Singh, Michael Wise (Miami University). arXiv: [2606.20898](https://arxiv.org/abs/2606.20898). First release: 2026-06-18
- Venue: **Preprint (arXiv).** No venue was found — [arXiv](https://arxiv.org/abs/2606.20898); [EmergentMind](https://www.emergentmind.com/papers/2606.20898)
- Korean summary:
  1. 제조 안전 교육 사례를 대상으로 한 문서 기반 생성형 AI 응용에서, 의미 검색 기반 RAG와 전체 문서 집합을 컨텍스트에 넣는 long-context 프롬프팅을 정확도와 토큰 비용 양쪽에서 비교했다.
  2. long-context가 정답률이 가장 높았지만(73.1% vs semantic RAG 65.4%), 질의당 토큰 비용은 약 26배였다. 저자들은 이 차이를 더 넓은 증거에 접근하는 대가인 '토큰 세(token tax)'로 정리했다.
- Quote: "73.1% vs. 65.4% for semantic RAG" (as returned by the search index) — [arXiv](https://arxiv.org/abs/2606.20898)
- Caveat: this is a single-domain case study. Secondary write-ups report fewer than 500 tokens per query for RAG versus more than 12,000 for long-context; that figure was not verified in the paper — [UBOS summary (secondary)](https://ubos.tech/the-token-tax-of-epistemic-accuracy-comparing-rag-and-long-context-architectures-for-document-grounded-generative-ai-applications/)

#### E. System efficiency (decoding acceleration, KV-cache reuse)

**[E1] REFRAG: Rethinking RAG based Decoding**
- Authors: Xiaoqiang Lin, Aritra Ghosh, Bryan Kian Hsiang Low, Anshumali Shrivastava, Vijai Mohan (Meta Superintelligence Labs / NUS / Rice). arXiv: [2509.01092](https://arxiv.org/abs/2509.01092). First release: 2025-09-01 (v2 2025-10-12)
- Venue: **NeurIPS 2025 Workshop poster** (Workshop on Multi-Turn Interactions in LLMs). No main-conference acceptance (e.g., ICLR 2026) was found, so the main-track status is **Unverified** — [NeurIPS 2025 virtual page](https://neurips.cc/virtual/2025/128022)
- Korean summary:
  1. RAG 컨텍스트에서는 청크 내부 어텐션이 강하고 청크 사이 어텐션은 약한 블록 대각(block-diagonal) 희소 구조가 나타난다. 이에 착안해 검색된 청크 대부분을 미리 계산해 둔 압축 청크 임베딩으로 바꿔 디코더에 직접 넣고, RL 정책이 일부 청크만 골라 원문 토큰으로 되돌리는 디코딩 프레임워크를 제안한다. LLM 구조를 바꾸거나 디코더 파라미터를 새로 추가하지 않는다.
  2. perplexity 손실 없이 TTFT를 30.85배 가속했고, 이는 기존 연구보다 3.75배 개선된 수치다.
- Quote: "30.85× the time-to-first-token acceleration (3.75× improvement to previous work) without loss in perplexity" — [arXiv](https://arxiv.org/abs/2509.01092)
- Conflict note: the arXiv HTML body gives 30.75× while the abstract gives 30.85×. Use the abstract figure — [arXiv HTML v1](https://arxiv.org/html/2509.01092v1)

**[E2] ProphetKV: User-Query-Driven Selective Recomputation for Efficient KV Cache Reuse in Retrieval-Augmented Generation**
- Authors: Shihao Wang, Jiahao Chen, Yanqi Pan, et al. arXiv: [2602.02579](https://arxiv.org/abs/2602.02579v3). First release: 2026-01-31 (revised in Feb)
- Venue: **ICML 2026 (Poster)**, PMLR vol. 306 — [PMLR](https://proceedings.mlr.press/v306/wang26y.html); [ICML 2026 virtual poster](https://icml.cc/virtual/2026/poster/63876)
- Korean summary:
  1. 미리 계산한 청크 KV 캐시를 이어 붙여 재사용하면 청크 사이 어텐션이 빠진다. 이를 보정하는 선택적 재계산에서, 기존 방법은 전역적으로 두드러지지만 질의와는 무관한 토큰에 재계산 예산을 써서 정확도가 떨어진다. ProphetKV는 사용자 질의와의 관련성을 기준으로 재계산할 토큰을 고른다.
  2. 재계산 비율 20%만으로 full-prefill 정확도의 96~101%를 유지했다. CacheBlend, EPIC, KVShare 같은 SOTA 방법보다 RULER에서 8.8~24.9%, LongBench에서 18.6~50.9% 정확도가 높았다.
- Quote: "retains 96%-101% of full-prefill accuracy with only a 20% recomputation ratio" — [arXiv HTML](https://arxiv.org/html/2602.02579)

#### Additional candidates surfaced (partially verified; for context, not in the core 12)
- **Stop Overthinking: Unlocking Efficient Listwise Reranking with Minimal Reasoning** (Danyang Liu, Kan Li; BIT; arXiv 2605.14450, 2026-05). It notes that reasoning-reranker latency "is dominated by the decoding time" and proposes Length-Regularized Self-Distillation. Venue: Preprint — [arXiv PDF](https://arxiv.org/pdf/2605.14450)
- **Efficient Listwise Reranking with Compressed Document Representations (RRK)** (arXiv 2604.26483, 2026-04). It compresses documents into fixed-size embeddings so that the reranker prompt holds only a few tokens per document. Venue not checked — [arXiv PDF](https://arxiv.org/pdf/2604.26483)
- **QRRanker** (arXiv 2602.12192). It trains query-focused retrieval heads as a listwise scorer. Venue not checked — [alphaXiv](https://www.alphaxiv.org/abs/2602.12192.md)
- **REARANK** (arXiv 2505.20046) is an RL-trained listwise reasoning reranker. A search result attributed it to EMNLP 2025, but the Anthology mapping was not confirmed — [arXiv PDF](https://arxiv.org/pdf/2505.20046)
- **Do Reasoning Models Enhance Embedding Models?** (arXiv 2601.21192, 2026-01) reports a null effect: embedders initialized from RLVR-tuned backbones show "no consistent performance advantage" on MTEB/BRIGHT — [arXiv PDF](https://arxiv.org/pdf/2601.21192)
- **CRE-T1 Preview** (arXiv 2603.17387) claims a 4B model matches the "retrieval + reranking" pipeline using only the original queries. It is a preview tech report — [arXiv HTML](https://arxiv.org/html/2603.17387)
- **GEM: A Generative Embedding Model Bridging Reasoning and Retrieval** (arXiv 2608.13200) and **Large Reasoning Embedding Models (LREM)** (arXiv 2510.14321). The titles were found; the abstracts were not verified — [GEM](https://arxiv.org/pdf/2608.13200); [LREM](https://arxiv.org/pdf/2510.14321)
- KV-cache-reuse follow-ups: **CacheClip** (2510.10129, v2 May 2026) uses an auxiliary model to choose tokens to recompute — [arXiv](https://arxiv.org/pdf/2510.10129). **From Prefix Cache to Fusion RAG Cache** (2601.12904) — [arXiv](https://arxiv.org/html/2601.12904v1). **RelaxKV** (2609.33503) does query-guided selective repair — [arXiv](https://arxiv.org/pdf/2609.33503). **ATTUNER** (2609.36722) is recomputation-free reuse via query-side adaptation; only the title was verified — [arXiv](https://arxiv.org/pdf/2609.36722)
- Pixel/optical compression siblings: **RAGOCR** (2608.00765) uses a GRPO-trained, query-aware optical compressor — [arXiv](https://arxiv.org/pdf/2608.00765). **PixelRAG** (2606.28344) does retrieval and reading over web screenshots and has an ICML 2026 virtual page — [ICML 2026](https://icml.cc/virtual/2026/80435); [arXiv](https://arxiv.org/abs/2606.28344)
- **The Attribution-Compression Frontier in RAG** (2609.14245) is an evaluation study. It finds that a RECOMP-style compressor's citations reach 0.86 precision against summaries but 0.12 against source spans on ASQA. The search reported it as a GroundLM 2026 workshop paper at EMNLP (unconfirmed) — [arXiv](https://arxiv.org/abs/2609.14245v1)

### Inferences
- The early-2025 foundational papers (ReasonIR, Rank1) worked because they **distilled reasoning-LLM traces into retrievers and rerankers**. Most of the 2025H2–2026 papers either scale this recipe (ReasonEmbed, ReasonRank → ACL 2026) or question it (LIMIT and *Rethinking Reasoning in Document Ranking* → ICLR 2026).
- Of the 12, 7 have verified 2026 main-venue acceptances (ICLR ×2, ACL ×2, WWW, ICML, NeurIPS). Industry-lab efficiency and compression papers (REFRAG/Meta, CLaRa/Apple) circulate widely without a confirmed main-venue acceptance.
- Verified venue acceptances suggest that ML venues (ICML/NeurIPS) now accept systems-flavored RAG efficiency work (ProphetKV, OpticalRAG), while IR/NLP venues (ACL, WWW) take retriever, reranker and compression methods.

### Gaps
- **ReasonIR venue**: only a secondary slide deck says "COLM 2025". Its acceptance could not be confirmed from COLM or OpenReview, so it stays "Unverified".
- **REFRAG main-venue status**: no ICLR 2026 / ICML 2026 decision was found. Only the NeurIPS 2025 workshop page is confirmed.
- **Exact first-release days** for LIMIT, ReasonEmbed, *Rethinking Reasoning in Document Ranking* and OpticalRAG could not be read from arXiv abs pages, because arxiv.org was blocked in-session. Only the months are given.
- **Re³Rank (arXiv 2611.02550, "Listwise Reranking with Reasoning-Conditioned Retrieval Heads")** appeared once as a search-index link title, with a claim that a 0.6B variant beats rerankers up to 32B. Two follow-up searches could not surface it, so it was **excluded as unverifiable**.
- Classic efficiency anchors (CacheBlend, Cache-Augmented Generation "CAG" arXiv 2412.15605, Provence arXiv 2501.16214, Speculative RAG) predate the window or were not re-verified this session, so they are not in the core list. Their venues should not be cited from these notes.
- Which 2026 preprints (CLaRa, Token Tax, Stop Overthinking) are under ICLR 2027 review is unknown. OpenReview was not reachable.

## Q2. How is this sub-area subdivided (retrieval quality vs. context utilization/compression vs. system efficiency vs. long-context debate)?

### Takeaway
Five sub-categories fit the evidence best. (A) Retrieval quality: reasoning-intensive and LLM-based embedders, plus the theoretical limits of single-vector embeddings. (B) Reranking: reasoning/test-time-compute rerankers and the efficiency backlash against them. (C) Context utilization: compression, pruning and joint retriever–generator training. (D) Long-context vs RAG. (E) System efficiency: decoding-side compression and KV-cache reuse. REFRAG and CLaRa sit across the C/E and C/A boundaries.

### Cited Findings
- **A. Retrieval quality (retrievers / embeddings)** — ReasonIR [A1], LIMIT [A2], ReasonEmbed [A3]. Extras: CRE-T1, GEM, LREM, *Do Reasoning Models Enhance Embedding Models?*
  - A1 sub-type, *reasoning-intensive embedders trained on synthetic hard queries and negatives*: ReasonIR's ReasonIR-Synthesizer — [HF Papers](https://huggingface.co/papers/2504.20595). ReasonEmbed's ReMixer and Redapter — [ACL Anthology](https://preview.aclanthology.org/ingest-acl/2026.acl-long.54/)
  - A2 sub-type, *capacity limits of single-vector retrieval*: LIMIT shows the embedding dimension bounds the set of returnable top-k subsets and suggests cross-encoders or multi-vector models as alternatives — [OpenReview PDF](https://openreview.net/pdf/43db7db7a837ff1e0af8065943aec2d036382459.pdf)
  - BRIGHT is the shared yardstick. ReasonIR reports 29.9 / 36.9 nDCG@10 (without / with reranker), ReasonEmbed reports 38.1 and ReasonRank reports 40.6 on the leaderboard — [HF Papers](https://huggingface.co/papers/2504.20595); [arXiv HTML ReasonEmbed](https://arxiv.org/html/2510.08252v2); [arXiv HTML ReasonRank](https://arxiv.org/html/2508.07050v1)
- **B. Reranking** — Rank1 [B1] (pointwise, test-time compute), ReasonRank [B2] (listwise, SFT→RL), *Rethinking Reasoning in Document Ranking* [B3] (critique). Extras: Stop Overthinking, RRK, QRRanker, REARANK.
  - Two directions: *scaling reasoning* (Rank1 distilled more than 600k R1 traces — [alphaXiv](https://www.alphaxiv.org/abs/2502.18418.md)) versus *making it cheaper or removing it* (CoT rerankers underperform direct ones — [ICLR 2026](https://iclr.cc/virtual/2026/poster/10006933); compressing reasoning traces — [arXiv 2605.14450](https://arxiv.org/pdf/2605.14450); compressing documents for the reranker — [arXiv 2604.26483](https://arxiv.org/pdf/2604.26483))
- **C. Context utilization: compression, pruning and joint training** — SeleCom [C1] (query-conditioned soft compression), CLaRa [C2] (latent compression with joint retriever–generator training), OpticalRAG [C3] (pixel-space compression). Extras: RAGOCR, PixelRAG, EnComp (2603.09222), edge adaptive compression (2608.19535)
  - Soft-compression critique and fix: query-agnostic full compression underperforms uncompressed RAG; the selector reframing fixes it — [arXiv](https://arxiv.org/abs/2602.15856)
  - Joint optimization: a differentiable top-k (Straight-Through) passes generator gradients to the retriever — [arXiv](https://arxiv.org/abs/2511.18659)
  - A 2026 compression survey listing found by search: *Contextual Compression in RAG for LLMs: A Survey* (arXiv 2409.13385) — [arXiv PDF](https://arxiv.org/pdf/2409.13385)
- **D. Long-context vs RAG** — Token Tax [D1]. Earlier evidence: LongBench v2 found RAG gave Qwen2.5 and GLM-4-Plus no gains beyond 32k retrieved tokens, while GPT-4o kept improving to 128k — [LongBench v2 arXiv](https://arxiv.org/pdf/2412.15204). *Long Context vs. RAG for LLMs: An Evaluation and Revisits* (2501.01880) reviews the conflicting findings — [arXiv PDF](https://arxiv.org/pdf/2501.01880). RAGSmith (2511.01386) reads the field as favoring a hybrid "RAG first, escalate when needed" approach — [arXiv PDF](https://arxiv.org/pdf/2511.01386)
- **E. System efficiency** — REFRAG [E1] (decoder-side chunk-embedding compression with RL-selected expansion), ProphetKV [E2] (query-driven selective KV recomputation). Extras: CacheClip, Fusion RAG Cache, RelaxKV, ATTUNER
  - Problem framing shared by E2 and its siblings: position-independent precomputed chunk caches lose cross-chunk attention, so methods decide which tokens to recompute — [arXiv RelaxKV](https://arxiv.org/pdf/2609.33503); [arXiv ProphetKV](https://arxiv.org/html/2602.02579)

### Inferences
- A useful two-axis framing for the report:
  - **Quality axis:** A (find the right evidence) → B (order it) → C (fit it into the generator).
  - **Cost axis:** C (fewer tokens) → E (cheaper tokens: KV reuse, decoding shortcuts) → D (whether to retrieve at all versus paying the long-context "token tax").
- Category C is turning into a hub. SeleCom, OpticalRAG and CLaRa all fuse "selection" (which used to be reranking) with "compression", and CLaRa also fuses in retriever training. The old boundary between retrieve → rerank → compress is blurring into a single query-conditioned context builder.

### Gaps
- No 2026 survey was found that formally proposes this exact five-way taxonomy. The taxonomy is my synthesis from the paper set, not a citation.
- "Chunking" (late chunking, mixture-of-chunkers) is in scope, but no 2026 chunking paper with a verified venue surfaced in this session's searches.
- Instruction-following retrievers and query rewriting/expansion (e.g., Promptriever, ThinkQE) were not verified this session. LIMIT's argument about instruction-based queries is the closest verified item.

## Q3. What new directions emerged in 2026?

### Takeaway
Five 2026 directions stand out: (1) **query-conditioned everything**, applying the query to compression and to KV recomputation; (2) **pixel/optical context compression**; (3) **an efficiency and correctness backlash against long-CoT reasoning rerankers**; (4) **maturing but contested reasoning-aware embedders**; and (5) **cost-explicit comparisons of long context vs RAG** (the "token tax").

### Cited Findings
- **Query-conditioned compression replaces query-agnostic soft compression.** SeleCom turns the compressor into a query-conditioned selector and matches or exceeds uncompressed RAG with 33.8–84.6% lower compute and latency (WWW 2026) — [arXiv](https://arxiv.org/abs/2602.15856). OpticalRAG's Query-Driven Distillation produces query-conditioned prefix tokens (NeurIPS 2026) — [NeurIPS 2026](https://neurips.cc/virtual/2026/poster/152209)
- **Query-driven KV-cache reuse.** ProphetKV chooses which tokens to recompute by relevance to the user query and beats CacheBlend, EPIC and KVShare (ICML 2026) — [PMLR](https://proceedings.mlr.press/v306/wang26y.html). RelaxKV (2609) and ATTUNER (2609, query-side adaptation) continue this line in late 2026 — [RelaxKV](https://arxiv.org/pdf/2609.33503); [ATTUNER](https://arxiv.org/pdf/2609.36722)
- **Pixel-space / optical compression of retrieved text.** OpticalRAG injects only 16 tokens into the LLM (NeurIPS 2026) — [NeurIPS 2026](https://neurips.cc/virtual/2026/poster/152209). RAGOCR (2608.00765) uses GRPO-trained query-aware optical compression — [arXiv](https://arxiv.org/pdf/2608.00765). PixelRAG (2606.28344) reads web screenshots and reports up to 3× token-cost reduction at lower resolutions — [arXiv](https://arxiv.org/abs/2606.28344); [ICML 2026 page](https://icml.cc/virtual/2026/80435). Zilliz frames this line as a DeepSeek-OCR-style "optical compression" idea for long-context and RAG (secondary) — [Zilliz blog](https://zilliz.com/blog/deepseek-ocr-explained-optical-compression-for-scalable-long-context-and-rag-systems)
- **Backlash against long-CoT reasoning rerankers.**
  - ICLR 2026: CoT rerankers "consistently underperform" direct rankers — [ICLR 2026](https://iclr.cc/virtual/2026/poster/10006933)
  - 2026-05 *Stop Overthinking*: extended reasoning yields diminishing returns, and latency is dominated by decoding — [arXiv](https://arxiv.org/pdf/2605.14450)
  - 2026-04 RRK: compressed document representations for listwise reranking — [arXiv](https://arxiv.org/pdf/2604.26483)
  - InvariRank (2604.27599) is a single-pass, permutation-invariant listwise reranker reported as accepted to SIGIR '26 (from the search summary, not confirmed on the SIGIR list; recommendation-framed) — [arXiv](https://arxiv.org/pdf/2604.27599)
- **Reasoning-aware embedders mature but are contested.**
  - ReasonEmbed is accepted to ACL 2026 with BRIGHT 38.1 — [ACL Anthology](https://preview.aclanthology.org/ingest-acl/2026.acl-long.54/)
  - CRE-T1 (2603) claims retriever-only parity with retrieve+rerank pipelines — [arXiv](https://arxiv.org/html/2603.17387)
  - A January 2026 study finds that RLVR-tuned backbones give "no consistent performance advantage" for embeddings — [arXiv](https://arxiv.org/pdf/2601.21192)
  - LIMIT (ICLR 2026) shows single-vector capacity limits, which pushes toward multi-vector, cross-encoder or generative alternatives — [ICLR 2026](https://iclr.cc/virtual/2026/poster/10007818)
- **Cost-explicit long-context vs RAG.** Long context wins on accuracy (73.1% vs 65.4%) at about 26× the token cost, which the authors call a "token tax" (2606.20898) — [arXiv](https://arxiv.org/abs/2606.20898)
- **Late-2025 direction that 2026 builds on: compression-native, jointly trained RAG.** CLaRa trains reranker and generator end to end in a shared continuous latent space — [arXiv](https://arxiv.org/abs/2511.18659). REFRAG feeds compressed chunk embeddings to the decoder for a 30.85× TTFT speedup — [arXiv](https://arxiv.org/abs/2509.01092)

### Inferences
- The unifying 2026 theme is **query-aware budget allocation**. Whether the budget is tokens (SeleCom, OpticalRAG), recomputation (ProphetKV), reasoning length (Stop Overthinking) or context length (Token Tax), the work decides where to spend compute given the query, instead of compressing or reasoning uniformly.
- The reranking line is moving from "more test-time compute" (Rank1, Feb 2025) to "just enough reasoning" (2026). That mirrors the general LLM "overthinking" literature.
- Pixel-space compression is the most novel 2026 idea that already has a top-venue acceptance (NeurIPS 2026). It is likely to be read as a bridge between text RAG and multimodal RAG; this is an inference, not a cited claim.

### Gaps
- I could not find a 2026 survey that names these trends explicitly. The trend list is synthesized from individual papers.
- Late-2026 items (RelaxKV, ATTUNER, GEM, Re³Rank) were found by title only, and their abstracts or venues could not be confirmed.
- No verified 2026 work on **speculative RAG** or **cache-augmented generation (CAG)** follow-ups surfaced in this session's searches. Whether those lines remain active in 2026 is unknown from these notes.
