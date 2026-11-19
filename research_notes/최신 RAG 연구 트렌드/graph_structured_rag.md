# Graph-based / Structured-Knowledge RAG & Memory-Augmented RAG (기준일: 2026-11-19)

> **검증 방법 및 한계 (먼저 읽기)**
> - 이 세션에서는 `arxiv.org`, `export.arxiv.org`, `api.semanticscholar.org`에 직접 접속(WebFetch/curl)할 수 없었다(네트워크 허용 목록 밖). 따라서 모든 서지·초록 정보는 **WebSearch가 띄워준 1차 페이지(arXiv abs/HTML, ACL Anthology, PMLR, NeurIPS/ICLR/ICML/AAAI 공식 페이지, 저자 GitHub)의 색인 내용**을 근거로 했다. 초록 원문 전체를 한 글자씩 대조하지는 못했다.
> - 베뉴(venue)는 1차 출처(학회 proceedings/공식 accepted list/ACL Anthology/arXiv Comments)에서 확인된 것만 표기했다. 확인 불가 시 "Preprint (arXiv)" 또는 "Unverified".
> - 인용구 신뢰도 표시: **[Q✓]** = 검색에 노출된 1차 페이지 텍스트에서 해당 문구가 그대로 확인됨, **[Q~]** = 검색 요약과 일치하는 핵심 문구이나 원문 글자 단위 대조는 못 함(arXiv 직접 열람 차단).
> - 최초 공개일: 일(day)까지 확인된 경우만 YYYY-MM-DD, 그 외에는 arXiv ID(YYMM)에서 확정되는 YYYY-MM으로 표기.

## Q1. 2025–2026년 가장 영향력 있고 최신인 Graph/Structured RAG 및 Memory-RAG 논문과 각 논문의 채택 학회는?

### Takeaway
Graph/structured RAG는 2024–25년 LightRAG·HippoRAG 2가 정의한 "LLM 추출 그래프 + PPR/이중레벨 검색" 패러다임에서, 2025 하반기–2026년에는 (i) 관계 추출을 줄인 경량 그래프(LinearRAG, AtomicRAG), (ii) 계층 집계/트리 인덱스(LeanRAG, Ψ-RAG), (iii) 그래프형 에이전트 메모리(A-MEM, MAGMA, ComoRAG), (iv) "그래프가 언제 도움이 되는가"를 묻는 실증 연구(GraphRAG-Bench, Does Memory Need Graphs?)로 분화했다. 주요 논문들은 ICML 2025, NeurIPS 2025, ICLR 2026, AAAI 2026, WWW 2026, ACL 2026 Main, ICML 2026 등 탑 베뉴에 채택되었다.

### Cited Findings

#### 큐레이션 목록 (13편; ★ = 2024말–2025초 기초(foundational) 논문)

**1. ★ LightRAG: Simple and Fast Retrieval-Augmented Generation** — Zirui Guo et al. (Guo, Xia, Yu, Ao, Huang; HKU)
- arXiv: [2410.05779](https://arxiv.org/abs/2410.05779) · 최초 공개: 2024-10 (v3: 2025-04-28, [alphaXiv](https://alphaxiv.org/abs/2410.05779v3))
- 베뉴: **EMNLP 2025 Findings** (ACL Anthology ID 2025.findings-emnlp.568, pp. 10746–10761) — [ACL Anthology](https://preview.aclanthology.org/new-sigs/2025.findings-emnlp.568/)
- 요약(KR):
  1. 텍스트 인덱싱·검색 과정에 그래프 구조를 통합하고, 구체적 엔티티·관계를 찾는 저수준 검색과 넓은 주제·테마를 찾는 고수준 검색을 결합한 이중 레벨(dual-level) 검색 및 그래프+벡터 결합 표현을 제안.
  2. 증분 업데이트 알고리즘으로 전체 인덱스를 재구축하지 않고 신규 데이터를 그래프에 반영할 수 있게 해, 경량·고속 GraphRAG의 대표 오픈소스 구현(HKUDS/LightRAG)이 됨.
- 인용구: "employs a dual-level retrieval system" [Q~] — [arXiv HTML](https://arxiv.org/html/2410.05779v3)

**2. ★ From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (HippoRAG 2)** — Bernal Jiménez Gutiérrez et al. (Shu, Qi, Zhou, Su; OSU)
- arXiv: [2502.14802](https://arxiv.org/abs/2502.14802v2) · 최초 공개: 2025-02
- 베뉴: **ICML 2025** (PMLR vol. 267, pp. 21497–21515) — [PMLR](https://proceedings.mlr.press/v267/gutierrez25a.html), [ICML poster](https://icml.cc/virtual/2025/poster/45585)
- 요약(KR):
  1. 벡터 검색에 의존하는 RAG가 인간 장기기억의 연상성·의미 이해 능력을 재현하지 못한다는 문제를 제기하고, HippoRAG의 Personalized PageRank 기반 그래프 검색에 패시지 수준 통합을 심화하고 LLM의 온라인 활용을 강화한 HippoRAG 2를 제안.
  2. 사실 기억·연상 기억 과제 전반에서 표준 RAG를 앞서며, 연상 기억 과제에서 최신 임베딩 모델 대비 약 7% 향상을 보고 — RAG를 "비모수적 연속학습(장기 메모리)"으로 재정의.
- 인용구: "paves the way for non-parametric continual learning for LLMs" [Q✓] — [ICML poster page](https://icml.cc/virtual/2025/poster/45585); 7% 수치 — [arXiv PDF](https://arxiv.org/pdf/2502.14802)

**3. ★ A-Mem: Agentic Memory for LLM Agents** — Wujiang Xu et al. (Liang, Mei, Gao, Tan, Zhang; Rutgers 등)
- arXiv: [2502.12110](https://arxiv.org/abs/2502.12110) · 최초 공개: 2025-02 (v11: 2025-10)
- 베뉴: **NeurIPS 2025** (Main Conference Track) — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/19909c36f51abc4856b4560aff3d36d6-Abstract-Conference.html), [poster 119020](https://neurips.cc/virtual/2025/poster/119020)
- 요약(KR):
  1. 고정된 메모리 구조·연산에 의존하는 기존 에이전트 메모리의 한계를 지적하고, Zettelkasten 원리에 따라 새 메모리마다 맥락 설명·키워드·태그를 갖는 구조화 노트를 만들고 기존 메모리와 동적으로 링크해 지식 네트워크를 형성하는 에이전틱 메모리를 제안.
  2. 새 기억이 들어오면 기존 기억의 표현을 갱신(메모리 진화)하며, 6개 파운데이션 모델에서 기존 SOTA 대비 개선을 보고 — "그래프형(링크드) 에이전트 메모리"의 대표 기준선.
- 인용구: "dynamic indexing and linking" [Q~] — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/19909c36f51abc4856b4560aff3d36d6-Abstract-Conference.html)
- 참고: Paper Digest가 "Most Influential NeurIPS 2025 Papers" 9위로 집계 — [Paper Digest](https://www.paperdigest.org/certificate/bcf73fa6d42f9428)

**4. When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation (GraphRAG-Bench)** — Zhishang Xiang et al.
- arXiv: [2506.05690](https://arxiv.org/abs/2506.05690v3) · 최초 공개: 2025-06-06 (v3: 2026-02)
- 베뉴: **ICLR 2026** — [ICLR 2026 proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/6c9e01d6cefbbf4cdd265032550e767f-Paper-Conference.pdf), 공식 repo에 "ICLR'26 accepted (2026-01-26)" 명시 — [GitHub](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)
- 요약(KR):
  1. GraphRAG가 실제 과제에서 vanilla RAG보다 자주 뒤처진다는 선행 관찰에서 출발해, 정보 밀도가 다른 코퍼스와 난이도가 점증하는 과제로 그래프 구축·검색·생성 전 과정을 평가하는 GraphRAG-Bench를 제안.
  2. 단순 사실 검색에서는 기본 RAG가 동등하거나 우세하고, 다중 홉 추론·맥락 요약 같은 복잡 과제에서 그래프의 이점이 나타난다는 "GraphRAG 사용 조건" 지침을 제시.
- 인용구: "hierarchical knowledge retrieval and deep contextual reasoning" [Q~] — [arXiv](https://arxiv.org/abs/2506.05690v3)

**5. LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval** — Yaoze Zhang et al. (Wu, Cai, Wang, Yan, Mao, Wang, Shi; Zhang·Wu 공동 1저자)
- arXiv: [2508.10391](https://arxiv.org/abs/2508.10391) · 최초 공개: 2025-08
- 베뉴: **AAAI 2026** (AAAI-26, vol. 40, pp. 34862–34869) — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/40789), [GitHub "accepted by AAAI-26"](https://github.com/razzzyz/leanrag)
- 요약(KR):
  1. 계층적 KG 요약 노드들이 서로 연결되지 않은 "의미의 섬(semantic islands)"이 되어 커뮤니티 간 추론이 어렵다는 문제를 지적하고, 엔티티 클러스터를 만들고 집계 요약 간 명시적 관계를 새로 구성하는 의미 집계 + 세밀한 엔티티에서 출발해 상향식으로 그래프를 순회하는 구조 유도 검색을 제안.
  2. 4개 QA 벤치마크에서 응답 품질을 개선하면서 검색 중복(redundancy)을 46% 줄였다고 보고.
- 인용구: "reducing 46% retrieval redundancy" [Q✓] — [arXiv](https://arxiv.org/abs/2508.10391)

**6. ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning** — 1저자 미확인(검색에서 저자 목록 미노출; Wang et al.로 기억되나 확인 필요) (SCUT, Tencent WeChat AI, Pazhou Lab)
- arXiv: [2508.10419](https://arxiv.org/pdf/2508.10419) · 최초 공개: 2025-08
- 베뉴: **AAAI 2026** (vol. 40, issue 39, pp. 33557–33565) — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/40644)
- 요약(KR):
  1. 200K+ 토큰 장편 서사 이해에서 단발성 검색 RAG가 상태(state)를 유지하지 못하는 문제에 대해, 추론이 막히면 탐색 질의를 생성하고 검색된 증거를 공유 메모리 풀에 통합하는 반복적 추론 사이클(동적 메모리 워크스페이스)을 제안.
  2. 4개 장문 서사 벤치마크에서 가장 강한 베이스라인 대비 최대 11%의 일관된 상대 향상을 보고.
- 인용구: "consistent relative gains up to 11% compared to the strongest baseline" [Q✓] — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/40644)

**7. GraphRAG-R1: Graph Retrieval-Augmented Generation with Process-Constrained Reinforcement Learning** — Chuanyue Yu et al. (Zhao, …, Ziwei Zhang)
- arXiv: [2507.23581](https://arxiv.org/abs/2507.23581v2) · 최초 공개: 2025-07 (v2: 2026-01-27)
- 베뉴: **WWW 2026** (arXiv Comments: "Accepted by the Web Conference 2026"; ACM DOI 10.1145/3774904.3792589) — [arXiv v2](https://arxiv.org/abs/2507.23581v2), [HF README](https://huggingface.co/yuchuanyue/GraphRAG-R1/blob/main/README.md)
- 요약(KR):
  1. 다중 홉 질문을 단계적으로 분해하고 필요 시 그래프/텍스트 검색 도구를 호출하도록 LLM을 강화학습(GRPO 변형)으로 훈련하는 그래프+에이전트 하이브리드로, rollout 검색 강화 GRPO·과정 제약 보상(PRA, Cost-Aware F1)·단계별 훈련·그래프-텍스트 혼합 검색을 결합.
  2. 과도/과소 검색을 억제하는 보상 설계로 GraphRAG를 "정적 파이프라인"에서 "학습된 검색 정책"으로 확장(HotpotQA·2Wiki·MuSiQue 어댑터 공개).
- 인용구: "hybrid graph-textual retrieval" [Q~] — [arXiv HTML](https://arxiv.org/html/2507.23581v2)
- 주의: RL 검색 에이전트 일반은 타 연구자 범위이므로 "그래프 특화" 사례로만 포함.

**8. LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora** — Luyao Zhuang et al. (Chen, Xiao, Zhou, Zhang, Chen, Zhang, Huang; PolyU)
- arXiv: [2510.10114](https://arxiv.org/abs/2510.10114v1) · 최초 공개: 2025-10-11 (v4: 2025-11)
- 베뉴: **ICLR 2026** — [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10007603), [PolyU 기관 리포지토리](https://ira.lib.polyu.edu.hk/handle/10397/119733)
- 요약(KR):
  1. 대규모 코퍼스에서 LLM 관계 추출이 만드는 노이즈·비용 문제를 겨냥해, 경량 엔티티 추출과 의미적 연결만으로 관계 없는(relation-free) 계층 그래프 "Tri-Graph"를 만들고, 국소 의미 브리징으로 엔티티를 활성화한 뒤 전역 집계(PPR)로 패시지를 순위화하는 2단계 검색을 제안.
  2. 4개 벤치마크(HotpotQA, 2Wiki, MuSiQue, 의료)에서 GraphRAG 베이스라인 대비 개선을 보고하며, 그래프 구축 비용을 코퍼스 크기에 선형으로 맞춤.
- 인용구: "relation-free hierarchical graph, termed Tri-Graph" [Q~] — [arXiv HTML](https://arxiv.org/html/2510.10114v4)
- 참고: "인덱싱 시간 77%+ 절감" 수치는 검색 요약에만 나오며 초록 원문 대조 못 함 — [arXiv PDF](https://arxiv.org/pdf/2510.10114)

**9. Does Memory Need Graphs? A Unified Framework and Empirical Analysis for Long-Term Dialog Memory** — Sen Hu et al. (Wei, Ran, Yao, Han, Wang, Chen, Lei Zou; PKU)
- arXiv: [2601.01280](https://arxiv.org/abs/2601.01280) · 최초 공개: 2026-01-03 (v3: 2026-02-10)
- 베뉴: **Unverified** — 다른 2026 논문의 참고문헌 목록이 "ACL 2026 long (2026.acl-long.1232)"로 인용했으나, ACL Anthology 원 페이지를 직접 확인하지 못함 — [인용 출처 arXiv 2609.08599](https://arxiv.org/pdf/2609.08599)
- 요약(KR):
  1. 그래프 구조가 대화 메모리 시스템에 널리 쓰이지만 효과에 대한 실증 결과가 엇갈린다는 문제의식에서, 메모리 추출·인덱싱·검색·QA 구성요소로 분해해 그래프/비그래프 방식을 모두 담는 통합 프레임워크를 만들고 LongMemEval·HaluMem에서 단계별 통제 실험을 수행.
  2. 성능 차이의 상당 부분이 특정 아키텍처 혁신보다 기초 시스템 설정에서 비롯됨을 보이고, 향후 연구를 위한 안정적인 강한 베이스라인을 제시.
- 인용구: "many performance differences are driven by foundational system settings rather than specific architectural innovations" [Q✓] — [arXiv](https://arxiv.org/abs/2601.01280)

**10. MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents** — Dongming Jiang et al. (Yi Li, Guanpeng Li, Bingzhe Li)
- arXiv: [2601.03236](https://arxiv.org/abs/2601.03236) · 최초 공개: 2026-01-06
- 베뉴: **ACL 2026 Main** (Long, 2026.acl-long.1709, pp. 36848–36865) — [ACL Anthology](https://aclanthology.org/2026.acl-long.1709/)
- 요약(KR):
  1. 기존 메모리 증강 LLM이 단일 메모리 저장소에서 의미 유사도로만 검색해 시간·인과·엔티티 정보가 뒤섞인다는 문제에 대해, 각 메모리 항목을 의미·시간·인과·엔티티의 직교 그래프에 나눠 저장하고 검색을 이 관계 뷰들 위의 정책 유도 순회로 정식화.
  2. 질의 적응적 뷰 선택과 투명한 추론 경로를 제공하며, LoCoMo·LongMemEval에서 최신 에이전틱 메모리 시스템을 능가했다고 보고.
- 인용구: "policy-guided traversal" [Q~] — [GitHub(초록 재수록)](https://github.com/FredJiang0324/MAGMA)

**11. STEM: Structure-Tracing Evidence Mining for Knowledge Graphs-Driven Retrieval-Augmented Generation** — Peng Yu et al. (En Xu, Bin Chen, Haibiao Chen, Yinfei Xu)
- arXiv: [2604.22282](https://arxiv.org/abs/2604.22282) · 최초 공개: 2026-04
- 베뉴: **ACL 2026 Main** (2026.acl-long.329; ACL 2026 Accepted Main Conference Papers 목록 등재) — [ACL Anthology](https://aclanthology.org/2026.acl-long.329/), [ACL 2026 accepted list](https://2026.aclweb.org/program/accepted_papers/)
- 요약(KR):
  1. KG 기반 RAG에서 그래프 구조 차이로 인한 의미 불일치와, 경로(path) 검색 방식의 전역 시야 부족 문제를 지적하고, 다중 홉 추론을 순차적 경로 탐색이 아닌 스키마 유도 서브그래프 매칭으로 재정식화; Triple-Dependent GNN으로 전역 가이드 서브그래프(구조적 사전지식)를 생성.
  2. 여러 다중 홉 벤치마크에서 정확도와 증거 서브그래프 완전성을 높여 SOTA를 보고.
- 인용구: "schema-guided subgraph matching" [Q~ — 검색 요약이 자체 표현으로 기술해 원문 일치 미확인] — [arXiv HTML](https://arxiv.org/html/2604.22282)

**12. AtomicRAG: Atom–Entity Graphs for Retrieval-Augmented Generation** — Yanning Hou et al. (외 7인)
- arXiv: [2604.20844](https://arxiv.org/abs/2604.20844) · 최초 공개: 2026-04
- 베뉴: **Preprint (arXiv)** — 채택 정보 미발견
- 요약(KR):
  1. 텍스트 청크를 기본 단위로 쓰는 GraphRAG가 여러 원자적 사실을 경직되게 묶고, 트리플 기반 엔티티 연결이 관계 추출 오류에 민감하다는 문제에 대해, 지식을 자기완결적 "지식 원자(atom)"로 저장하고 포함·관련성·동의어 엣지를 갖는 무라벨 가중 Atom-Entity Graph를 구축.
  2. 질의 분해 + 밀집 원자 검색 + 엔티티 시드 PPR + "Atomic Sieve" 필터로 압축된 컨텍스트를 구성해 검색 정확도와 추론 견고성 개선을 주장(구축 품질 의존성은 한계로 명시).
- 인용구: "rigidly group multiple atomic facts together" [Q~] — [arXiv HTML](https://arxiv.org/html/2604.20844v1), [GitHub](https://github.com/7HHHHH/AtomicRAG)

**13. Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation (Ψ-RAG)** — Ziwen Zhao, Menglin Yang
- arXiv: [2605.00529](https://arxiv.org/abs/2605.00529) · 최초 공개: 2026-05 (GitHub 기준 2026-05-04)
- 베뉴: **ICML 2026** — [ICML 2026 poster 66483](https://icml.cc/virtual/2026/poster/66483), [GitHub "[ICML'26]"](https://github.com/Newiz430/Psi-RAG)
- 요약(KR):
  1. RAPTOR류 단일 문서용 트리 RAG가 k-means 군집 노이즈, 문서 간 명시적 연결 부재, 거친 추상화로 문서 간(cross-document) 다중 홉 질의에 약하다는 문제에 대해, 청크 유사도 순위화와 반복적 병합·붕괴(merge-and-collapse)로 만든 계층적 추상 트리 인덱스와 다중 입도 에이전틱 검색기를 결합.
  2. 문서 간 다중 홉 QA에서 평균 F1 기준 RAPTOR 대비 25.9%, HippoRAG 2 대비 7.4% 향상을 보고하며 토큰 수준 QA부터 문서 수준 요약까지 지원.
- 인용구: "merging and collapse" [Q~] — [arXiv](https://arxiv.org/abs/2605.00529)

#### 보조 후보 (범위 내이나 본 목록에서 제외/보완용)
- **HyperGraphRAG** (Haoran Luo et al., [2503.21322](https://arxiv.org/abs/2503.21322), 2025-03) — **NeurIPS 2025** (Main) — n-ary 관계를 하이퍼엣지로 표현; 의료·농업·CS·법률에서 표준 RAG 및 기존 그래프 RAG 대비 정확도·검색 효율·생성 품질 개선 보고 — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/df55ee6e59f8ac4a625219e11fe9ddba-Abstract-Conference.html)
- **RAG vs. GraphRAG: A Systematic Evaluation and Key Insights** (Haoyu Han et al., [2502.11371](https://arxiv.org/abs/2502.11371), 2025-02-17; v3 2026-03-04) — **Preprint (arXiv)** (v3 PDF의 ACM 템플릿 placeholder는 채택 증거 아님) — [arXiv v3](https://arxiv.org/pdf/2502.11371v3)
- **Graph-R1: Towards Agentic GraphRAG Framework via End-to-end RL** (Haoran Luo et al., [2507.21892](https://arxiv.org/abs/2507.21892v1), 2025-07-29; v2 2026-06-02) — **Unverified** (ICML 스타일 템플릿만 확인, 채택 증거 없음) — [alphaXiv](https://alphaxiv.org/abs/2507.21892)
- **Deep GraphRAG** (Yuejie Li et al., Ant Group, [2601.11144](https://arxiv.org/abs/2601.11144v1), 2026-01-16) — **Preprint (arXiv)** — 커뮤니티 간 필터링→커뮤니티 정제→엔티티 검색의 3단계 계층 검색 + 빔서치 동적 리랭커 + 보상가중 GRPO로 학습한 소형(1.5B) 통합 모델, NQ·HotpotQA 평가 — [arXiv HTML](https://arxiv.org/html/2601.11144v1)
- **ZoomRAG** (계층적 랜덤워크 "줌" 검색) — **ACL 2026 Findings** (2026.findings-acl.1643) — arXiv ID 미확인 — [ACL Anthology PDF](https://aclanthology.org/2026.findings-acl.1643.pdf)
- **Token-Free Hierarchical Indexing for RAG beyond LLM-based Summarization (SeRAG)** — **ICML 2026** — LLM 재귀 요약 대신 정보이론적 분류체계로 계층 인덱스 구축 — arXiv ID 미확인 — [ICML poster 63668](https://icml.cc/virtual/2026/poster/63668)
- **Retrieval-Augmented Generation with Retrieval-optimized TRA Structure** (JungHyoun Kim et al.) — **EMNLP 2026 Main** (2026.emnlp-main.118) — Title-Relations-Attributes 데이터베이스 구조 제안(구조 인식 RAG) — [ACL Anthology](https://aclanthology.org/2026.emnlp-main.118/)
- **Temporal Semantic Memory for Personalized LLM Agents** — **ACL 2026 Findings** (2026.findings-acl.1496) — [ACL Anthology PDF](https://aclanthology.org/2026.findings-acl.1496.pdf)
- 기타 2026 그래프 RAG 프리프린트: GraFine ([2601.18579](https://arxiv.org/pdf/2601.18579), 검색 시점 정제), AGRAG ([2511.05549](https://arxiv.org/html/2511.05549), TF-IDF 기반 그래프 구축으로 LLM 추출 환각 감소), EraRAG ([2506.20963](https://arxiv.org/pdf/2506.20963), 성장 코퍼스용 증분 계층), LivingRAG ([2608.25960](https://arxiv.org/pdf/2608.25960), 경험으로 Graph RAG 증강), SVD-RAG ([2607.10316](https://arxiv.org/pdf/2607.10316), RAPTOR 트리에서 LLM 요약을 SVD 추출 요약으로 대체) — 모두 베뉴 미확인.

### Inferences
- 2025→2026 탑 베뉴 채택 흐름: ICML 2025(HippoRAG 2) → NeurIPS 2025(HyperGraphRAG, A-Mem) → AAAI 2026(LeanRAG, ComoRAG) → ICLR 2026(LinearRAG, GraphRAG-Bench) → WWW 2026(GraphRAG-R1) → ACL 2026 Main(MAGMA, STEM) → ICML 2026(Ψ-RAG, SeRAG) → EMNLP 2026(TRA). 그래프/구조 RAG는 ML·NLP·웹/IR 베뉴 전반에서 꾸준히 채택되는 주류 하위 분야로 볼 수 있다.
- HippoRAG 2(PPR over passage/phrase graph)는 LinearRAG·AtomicRAG의 PPR 기반 검색, Ψ-RAG의 비교 기준선으로 계속 등장 → 2026년 "사실상의 강한 그래프 베이스라인"으로 기능.
- LightRAG의 영향력은 학회 등급(Findings)보다 오픈소스 채택에서 비롯된 것으로 보인다(추정; 정량 근거는 미수집).

### Gaps
- arXiv abs 페이지(Comments 필드)를 직접 열람하지 못해, 일부 논문의 정확한 v1 일자와 초록 원문 대조가 불완전함(특히 [Q~] 표시 인용구).
- "Does Memory Need Graphs?"의 ACL 2026 채택 여부는 2차 인용(참고문헌 목록)만 있고 ACL Anthology 원 엔트리로 확인 못함.
- ComoRAG 1저자 이름을 1차 출처로 확인하지 못함.
- NodeRAG(2504.11544), PathRAG(2502.14902), Mem0(2504.19413), Zep(2501.13956), GFM-RAG(2502.01113), ArchRAG(2502.09891)는 시간 제약으로 베뉴 확인을 하지 못해 목록에서 제외(누락이 중요도 판단을 의미하지 않음).

## Q2. 이 하위 분야는 어떻게 세분되는가?

### Takeaway
대표 서베이는 파이프라인 단계(그래프 인덱싱 / 그래프 유도 검색 / 그래프 강화 생성)로 분류하지만, 2025–26년 논문 흐름은 "무엇을 그래프로 만들고 어떻게 순회하는가" 기준의 6개 하위 범주로 나누는 것이 실용적이다: (A) KG 구축형 GraphRAG, (B) 경량/관계-free 그래프 인덱싱, (C) 계층 요약 트리·커뮤니티 계층, (D) 메모리-as-그래프(에이전트 장기기억), (E) 그래프+에이전트 추론(KG 추론/RL), (F) 실증 분석·벤치마크.

### Cited Findings
- 서베이 분류 1: "Graph Retrieval-Augmented Generation: A Survey"(Peng et al., 2408.08921)는 Graph-based Indexing(G-Indexing), Graph-guided Retrieval(G-Retrieval), Graph-enhanced Generation(G-Generation)의 3단계로 정리 — [arXiv](https://arxiv.org/pdf/2408.08921). (ACM TOIS 게재 여부는 이번 검색에서 확인 못함)
- 서베이 분류 2: "Retrieval-Augmented Generation with Graphs (GraphRAG)"(Han et al., 2501.00309)는 query processor, retriever, organizer, generator, data source의 5요소 프레임워크로 정리 — [arXiv](https://arxiv.org/pdf/2501.00309)
- 서베이 분류 3: Zhu, Huang et al.(2504.10499)은 데이터베이스 구축·알고리즘·파이프라인·태스크 관점의 분류 제시(2026-01 템플릿 개정본) — [arXiv](https://arxiv.org/pdf/2504.10499v2)
- 에이전트 메모리 쪽: "Does Memory Need Graphs?"는 대화 메모리를 메모리 추출·인덱싱·검색·QA 4단계로 분해 — [arXiv](https://arxiv.org/abs/2601.01280); 2026년 그래프 기반 개인화 메모리 서베이(표현·진화·검색·평가) 존재 — [arXiv 2609.08599](https://arxiv.org/pdf/2609.08599)

#### 제안 하위 범주와 논문 배치
| 하위 범주 | 핵심 아이디어 | 해당 논문 (본 목록 / 보조) |
|---|---|---|
| (A) KG/엔티티-관계 그래프 구축형 GraphRAG | LLM으로 엔티티·관계(또는 n-ary) 추출 → 그래프/KG 위에서 검색 | LightRAG★, STEM(KG-driven, 서브그래프 매칭) / HyperGraphRAG, MegaRAG(멀티모달, 범위 외) |
| (B) 경량·관계-free·원자 단위 그래프 인덱싱 | 관계 추출 비용·노이즈 제거, PPR 기반 확산 | HippoRAG 2★, LinearRAG, AtomicRAG / AGRAG, GraFine |
| (C) 계층 요약 트리·커뮤니티 계층(RAPTOR/MS GraphRAG 계열) | 군집·요약으로 다층 인덱스, 상향/하향 순회 | LeanRAG, Ψ-RAG / Deep GraphRAG, ZoomRAG, SeRAG, SVD-RAG, EraRAG |
| (C') 구조 인식 RAG(표·DB·문서 구조) | 데이터 고유 구조를 인덱스로 사용 | / TRA structure(EMNLP 2026), Semantic Compression Trees(문서 경계 기반) |
| (D) 메모리-as-그래프 / 메모리 증강 RAG | 에이전트·장문 상호작용 기억을 링크/다중 그래프로 저장, 검색=순회 | A-Mem★, MAGMA, ComoRAG, HippoRAG 2★(가교) / Temporal Semantic Memory, TiMem·APEX-MEM·MemORAI(2차 인용만) |
| (E) 그래프+에이전트 추론(KG 추론 에이전트, RL) | LLM이 그래프 탐색을 정책으로 학습/수행 | GraphRAG-R1, MAGMA(정책 유도 순회), Ψ-RAG(에이전틱 검색기) / Graph-R1, Deep GraphRAG |
| (F) 실증 분석·벤치마크 | "그래프가 언제 도움이 되는가", 통제 비교 | GraphRAG-Bench, Does Memory Need Graphs? / RAG vs. GraphRAG |

(근거: 각 논문 항목의 Q1 출처 참조. Semantic Compression Trees — [arXiv 2608.21610](https://arxiv.org/pdf/2608.21610))

### Inferences
- (B)와 (C)는 모두 "MS GraphRAG/LightRAG식 LLM 관계 추출이 비싸고 노이즈가 많다"는 공통 비판에서 출발해 서로 다른 방향(평평하고 싼 그래프 vs. 더 잘 연결된 계층)으로 해결하고 있다.
- (D)와 (E)의 경계가 2026년에 흐려지는 중: MAGMA는 메모리 구조(D)이면서 검색을 정책 순회(E)로 정의하고, Ψ-RAG는 트리 인덱스(C)에 에이전틱 검색기(E)를 결합.
- (F)는 별도 범주로 둘 가치가 있음: 2025–26년 실증 연구들이 (A)–(D) 각 범주의 가치 주장을 직접 반박/조건부화하고 있기 때문.

### Gaps
- 위 6범주 분류는 서베이의 공식 분류가 아니라 수집 논문을 바탕으로 한 제안이다. 2026년 최신 GraphRAG 서베이(분류 체계)를 1차 확인하지 못함.
- 구조 인식 RAG(표·문서 구조) 분야는 이번 조사에서 대표 논문을 충분히 확보하지 못함(TRA 논문 초록 전문도 미확인).

## Q3. 비판적·실증적 연구는 GraphRAG가 언제 vanilla RAG를 이기며, 비용은 어떻다고 말하는가?

### Takeaway
실증 연구들의 공통 결론은 "그래프는 조건부로만 이득"이다. 단일 홉·국소 사실 검색에서는 vanilla RAG가 동등하거나 우세하고, 다중 홉 추론·전역 요약·연상 기억에서 그래프가 이득을 보인다. 에이전트 메모리에서도 성능 차이의 상당 부분이 그래프 구조 자체보다 기초 설정에서 온다는 결과가 나왔다. 비용 측면에서는 LLM 기반 그래프 구축·긴 프롬프트의 "토큰 세금"이 핵심 약점이며, 2025–26년 방법들(LinearRAG, LeanRAG, SeRAG)은 이를 줄이는 방향으로 움직이고 있다.

### Cited Findings
- GraphRAG-Bench(ICLR 2026)는 선행 연구에서 GraphRAG가 실제 과제에서 vanilla RAG보다 자주 뒤처진다고 보고된 점을 동기로 삼음 — [arXiv](https://arxiv.org/abs/2506.05690v3)
- GraphRAG-Bench 인트로가 인용한 선행 결과: Natural Questions에서 GraphRAG 정확도가 vanilla RAG보다 13.4% 낮음(GraphRAG-Bench 자체 실험 수치가 아님) — [arXiv HTML](https://arxiv.org/html/2506.05690v1)
- GraphRAG-Bench 결론: 단순 사실 검색에서는 기본 RAG가 동등/우세하고 그래프가 중복·노이즈를 주입할 수 있음; 복잡한 다중 홉 추론에서 그래프의 이점이 나타남 — [arXiv](https://arxiv.org/abs/2506.05690v3), [GitHub](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)
- 2차 출처 수치(검증 필요): 복잡 추론 53.4 vs 42.9, 맥락 요약 64.4 vs 51.3(그래프 vs 텍스트 청크); 의료 요약에서는 reranked RAG(65.75%)가 HippoRAG2(63.08%)를 근소하게 앞섬 — [papernotes 요약](https://en.papernotes.org/ICLR2026/information_retrieval/when_to_use_graphs_in_rag_a_comprehensive_analysis_for_graph_retrieval-augmented/), [VentureBeat](https://venturebeat.com/orchestration/stop-graphing-everything-when-graphrag-actually-beats-vector-rag) (2차 출처; 원문 표 대조 못함)
- RAG vs. GraphRAG(Han et al.): RAG와 GraphRAG는 일관된 승자 없이 상호보완적이며, QA에서 RAG는 단일 홉, GraphRAG는 추론 중심 질문에 강함 — [arXiv v3](https://arxiv.org/pdf/2502.11371v3)
- Does Memory Need Graphs?: 대화 메모리에서 성능 차이의 상당 부분이 특정 아키텍처 혁신보다 기초 시스템 설정에서 비롯됨 — [arXiv](https://arxiv.org/abs/2601.01280)
- 비용 개선 사례: LeanRAG는 검색 중복 46% 감소 — [arXiv](https://arxiv.org/abs/2508.10391); LinearRAG는 관계 추출을 제거한 Tri-Graph로 구축 비용 절감(인덱싱 시간 77%+ 절감은 검색 요약 기준, 원문 대조 필요) — [arXiv PDF](https://arxiv.org/pdf/2510.10114); SeRAG(ICML 2026)는 재귀적 LLM 요약의 높은 비용과 정보 손실을 문제로 지목 — [ICML poster](https://icml.cc/virtual/2026/poster/63668)
- 트리(RAPTOR) 계열에 대한 반례: TreeHop 논문은 top-5 RAPTOR가 여러 벤치마크에서 직접 검색보다 뒤처진다고 보고 — [arXiv 2504.20114](https://arxiv.org/pdf/2504.20114) (검색 요약 기준)
- 반대로 Ψ-RAG는 문서 간 다중 홉 QA에서 RAPTOR 대비 F1 +25.9%, HippoRAG 2 대비 +7.4%를 보고(트리 구조가 문서 간 연결을 갖추면 이득) — [arXiv](https://arxiv.org/abs/2605.00529)

### Inferences
- 실무 판단 규칙(추론): 질의가 국소 사실 조회 위주면 vanilla RAG(+리랭커)를, 다중 홉·전역 요약·장기 연상 기억이 핵심이면 그래프/계층 인덱스를 고려. 그래프 도입 전 "강한 평면 베이스라인"과 통제 비교가 필수라는 점이 GraphRAG-Bench와 Does Memory Need Graphs?에서 공통적으로 도출됨.
- 2026년 신규 방법들이 정확도뿐 아니라 "구축 비용·중복 감소"를 헤드라인 수치로 내세우는 것은, 실증 연구들이 제기한 비용 비판에 대한 대응으로 해석 가능.

### Gaps
- GraphRAG-Bench의 토큰 비용 수치("token tax")를 원문 표로 확인하지 못함(2차 요약에서만 언급).
- "LightRAG·HippoRAG가 증거 재현율은 다소 높이지만 컨텍스트 관련도는 vanilla RAG보다 크게 낮다"는 결과가 검색 요약에 등장했으나 출처 논문을 특정하지 못해 인용에서 제외.
- "In-depth Analysis of Graph-based RAG in a Unified Framework"(2503.04338)와 GraphRAG-Bench(Xiao et al., 2506.02404, 동명의 별도 벤치마크)는 확인하지 못함 — 이름이 같은 두 "GraphRAG-Bench"가 존재하므로 보고서 작성 시 혼동 주의.

## Q4. 2026년의 새로운 방향은?

### Takeaway
2026년 신호는 (1) 관계 추출을 버리거나 단위를 바꾼 그래프(지식 원자, relation-free), (2) 토큰-프리/저비용 계층 인덱싱과 문서 간 트리, (3) 다중 그래프(의미·시간·인과·엔티티) 에이전트 메모리와 시간적 메모리, (4) 그래프 검색을 정책으로 학습하는 에이전트화, (5) 연합/분산 GraphRAG, (6) 그래프 메모리 효과에 대한 회의적 실증 분석이다.

### Cited Findings
- 지식 원자 그래프: AtomicRAG(2026-04)는 청크 대신 자기완결적 지식 원자와 Atom-Entity Graph를 사용 — [arXiv HTML](https://arxiv.org/html/2604.20844v1)
- 관계-free 그래프: LinearRAG(ICLR 2026) — [ICLR poster](https://iclr.cc/virtual/2026/poster/10007603)
- 저비용·문서 간 계층 인덱스: Ψ-RAG(ICML 2026) — [arXiv](https://arxiv.org/abs/2605.00529); SeRAG(ICML 2026, LLM 요약 없는 계층 인덱싱) — [ICML poster](https://icml.cc/virtual/2026/poster/63668); ZoomRAG(ACL 2026 Findings) — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1643.pdf)
- 다중 그래프 에이전트 메모리: MAGMA(ACL 2026 Main) — [ACL Anthology](https://aclanthology.org/2026.acl-long.1709/); 시간 그래프 개인화 메모리(ACL 2026 Findings) — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1496.pdf)
- 2026 하반기 메모리-그래프 프리프린트 다수(베뉴 미확인): Entity-Memory Graph Retrieval for long-conversation QA([2608.27925](https://arxiv.org/pdf/2608.27925)), RippleMem(연상적 회상, [2608.13334](https://arxiv.org/pdf/2608.13334)), Profile-Graph Memory([2607.19359](https://arxiv.org/pdf/2607.19359)), Event-Centric Memory with Query-Aware Graph Augmentation([2610.11920](https://arxiv.org/pdf/2610.11920)), SGMem(문장 그래프 메모리, [2509.21212](https://arxiv.org/pdf/2509.21212))
- 그래프 검색의 정책 학습: GraphRAG-R1(WWW 2026) — [arXiv](https://arxiv.org/abs/2507.23581v2); Deep GraphRAG(보상가중 GRPO로 소형 통합 모델 학습) — [arXiv HTML](https://arxiv.org/html/2601.11144v1); ProGraph-R1(2601.17755) 등장 — [fugumt 색인](https://fugumt.com/fugumt/paper_check/2601.17755v1_enmode)
- 연합/분산 GraphRAG: "Federated Graph Retrieval-Augmented Generation"(arXiv 2610.30728, 2026-10)은 클라이언트별로 분산된 KG를 경량 인덱스로 라우팅하고 통제된 공개(controlled disclosure)로 증거를 구성한다고 검색 요약에 나타남 — [arXiv](https://arxiv.org/abs/2610.30728) (단일 검색 요약 근거, 후속 검색에서 재확인 실패); SCOUT-RAG(분산 도메인 에이전틱 그래프 RAG, [2602.08400](https://arxiv.org/pdf/2602.08400v1))
- 경험 증강: LivingRAG — "Augmenting Graph RAG with Experience" — [arXiv 2608.25960](https://arxiv.org/pdf/2608.25960)
- (범위 외지만 신호) GraphRAG 보안: LogicPoison(그래프 RAG 대상 논리 공격)이 ACL 2026 Main 목록에 등재 — [ACL 2026 accepted list](https://2026.aclweb.org/program/accepted_papers/); 멀티모달 GraphRAG(MegaRAG, ACL 2026 long; HVM-GraphRAG) — [ACL Anthology](https://aclanthology.org/2026.acl-long.2218/)

### Inferences
- 2026년의 무게중심은 "더 큰 KG를 더 정교하게 추출"이 아니라 "더 싸고, 더 세밀한 단위로, 더 잘 연결된 인덱스 + 학습된 순회 정책"으로 이동 중이다.
- 에이전트 메모리 연구가 GraphRAG 기법(PPR, 엔티티 그래프, 계층 통합)을 흡수하면서 "memory-augmented RAG"와 "GraphRAG"가 사실상 하나의 설계 공간으로 수렴하고 있다(MAGMA, HippoRAG 2, ComoRAG).
- 연합/분산 GraphRAG와 그래프 대상 공격(LogicPoison) 등장은 GraphRAG가 엔터프라이즈 배포 단계로 진입했음을 시사(추정).

### Gaps
- 2026 하반기(7–11월) 프리프린트 다수는 초록 원문과 베뉴를 확인하지 못함; NeurIPS 2026 채택 결과(그래프 RAG 관련)를 확인하지 못함.
- FedGRAG(2610.30728)의 저자·초록 상세는 첫 검색 요약에만 나타나 재확인 실패 — 인용 시 주의.
- ICLR 2027 제출작은 심사 중이므로 베뉴 표기 대상 아님.
