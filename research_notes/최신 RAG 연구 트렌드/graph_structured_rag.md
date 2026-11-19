# Graph-based / Structured-Knowledge RAG & Memory-Augmented RAG (기준일: 2026-11-19)

> **검증 방법 및 한계 (먼저 읽기) — 2026-11-19 1차 출처 재검증판**
> - 초판은 arXiv·ACL Anthology·OpenReview에 접속하지 못해 검색 스니펫에 의존했다. 이번 재검증에서는 **arXiv abs 페이지(`arxiv.org/abs/<id>`: 제목·저자·Submission history·Comments·초록)를 모든 arXiv 논문에 대해 직접 내려받아 대조**했고, 필요 시 arXiv PDF 1면 머리말(학회 스타일 문구)도 확인했다. ACL 계열 베뉴는 **`aclanthology.org/<Anthology ID>/` 페이지의 citation 메타데이터(권 제목·쪽수·DOI·초록)**로 확인했다.
> - 접속 불가(재검증 시점): `api2.openreview.net`(프록시 403), `openreview.net`(브라우저 검증 페이지), `api.openreview.net`(v1 검색은 2026 학회 v2 노트를 반환하지 않고 `/notes` 질의는 challenge 403), `iclr.cc`, `icml.cc`, `neurips.cc`, `proceedings.iclr.cc`, `proceedings.neurips.cc`, `proceedings.mlr.press`, `ojs.aaai.org`, `doi.org`, `2026.aclweb.org`, `mlanthology.org`. 따라서 ICLR/ICML/NeurIPS/AAAI 베뉴는 **arXiv Comments 필드 + arXiv PDF 머리말(예: "Published as a conference paper at ICLR 2026", "Proceedings of the 43rd ICML … PMLR 306, 2026")**을 1차 근거로 삼았다. 이 경로로도 확인되지 않으면 ⚠️로 표시했다.
> - 인용구 신뢰도 표시: **[Q✓]** = arXiv abs 또는 ACL Anthology 초록 원문에서 글자 그대로 확인됨. 이번 재검증에서 모든 인용구를 원문과 대조했다.
> - 최초 공개일 = arXiv Submission history의 v1 날짜(YYYY-MM-DD). arXiv ID의 YYMM과 v1 날짜가 다른 경우(모더레이션 보류 등)는 따로 적었다.
> - 저자 소속(HKU, PolyU 등)은 arXiv abs 페이지에 나오지 않아 대조하지 않았다(초판 기재값 유지, 미검증).

## Q1. 2025–2026년 가장 영향력 있고 최신인 Graph/Structured RAG 및 Memory-RAG 논문과 각 논문의 채택 학회는?

### Takeaway
Graph/structured RAG는 2024–25년 LightRAG·HippoRAG 2가 정의한 "LLM 추출 그래프 + PPR/이중 레벨 검색" 패러다임에서, 2025 하반기–2026년에는 (i) 관계 추출을 줄인 경량 그래프(LinearRAG, AtomicRAG), (ii) 계층 집계/트리 인덱스(LeanRAG, Ψ-RAG), (iii) 그래프형 에이전트 메모리(A-MEM, MAGMA, ComoRAG), (iv) "그래프가 언제 도움이 되는가"를 묻는 실증 연구(GraphRAG-Bench, Does Memory Need Graphs?)로 분화했다. 주요 논문들은 EMNLP 2025 Findings, ICML 2025, NeurIPS 2025, AAAI 2026, ICLR 2026, WWW 2026, ACL 2026 Main, ICML 2026 등에 채택되었다(각 항목의 "검증" 줄 참조).

### Cited Findings

#### 큐레이션 목록 (13편; ★ = 2024말–2025초 기초(foundational) 논문)

**1. ★ LightRAG: Simple and Fast Retrieval-Augmented Generation** — Zirui Guo et al. (Guo, Xia, Yu, Ao, Huang; HKU)
- arXiv: [2410.05779](https://arxiv.org/abs/2410.05779) · 최초 공개: 2024-10-08 (v2: 2024-11-07, v3: 2025-04-28)
- 베뉴: **EMNLP 2025 Findings** (Findings of the ACL: EMNLP 2025, pp. 10746–10761, DOI 10.18653/v1/2025.findings-emnlp.568) — [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.568/)
- 요약(KR):
  1. 기존 RAG가 평면적(flat) 데이터 표현과 부족한 맥락 인식 때문에 복잡한 상호의존성을 놓친 단편적 답을 낸다는 문제에 대해, 텍스트 인덱싱·검색에 그래프 구조를 통합하고 저수준·고수준 지식 발견을 함께 하는 이중 레벨(dual-level) 검색 시스템을 제안.
  2. 그래프와 벡터 표현을 결합해 관련 엔티티·관계를 효율적으로 검색해 응답 시간을 줄이고, 증분 업데이트 알고리즘으로 새 데이터를 신속히 반영한다. 기존 방법 대비 검색 정확도·효율 개선을 보고하고 코드를 공개했다.
- 인용구: "employs a dual-level retrieval system" [Q✓] — [arXiv abs](https://arxiv.org/abs/2410.05779)
- 검증: ✅ arXiv abs 확인(제목·저자 5인·v1 2024-10-08·초록 일치; arXiv Comments 없음) / venue 출처: ACL Anthology 2025.findings-emnlp.568 메타데이터("Findings of the Association for Computational Linguistics: EMNLP 2025")

**2. ★ From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (HippoRAG 2)** — Bernal Jiménez Gutiérrez et al. (Shu, Qi, Zhou, Su; OSU)
- arXiv: [2502.14802](https://arxiv.org/abs/2502.14802) · 최초 공개: 2025-02-20 (v2: 2025-06-19)
- 베뉴: **ICML 2025** — arXiv Comments "ICML 2025"; (초판 인용, 이번에 재접속 불가) [PMLR vol. 267](https://proceedings.mlr.press/v267/gutierrez25a.html), [ICML poster](https://icml.cc/virtual/2025/poster/45585)
- 요약(KR):
  1. 벡터 검색에 의존하는 RAG는 인간 장기기억의 동적·상호연결적 성질을 모방하기 어렵다. 여기에 KG 등 구조를 덧붙인 최근 RAG는 의미 파악(sense-making)·연상성은 보완하지만 기본적인 사실 기억 과제에서는 표준 RAG보다 성능이 크게 떨어진다. 이 의도치 않은 성능 저하를 해결하려고, HippoRAG의 Personalized PageRank에 더 깊은 패시지 통합과 더 효과적인 온라인 LLM 활용을 더한 HippoRAG 2를 제안.
  2. 사실 기억·의미 파악·연상 기억 과제 전반에서 표준 RAG를 포괄적으로 앞서고, 연상 기억 과제에서 최신 임베딩 모델 대비 7% 향상을 보고해 LLM의 비모수적 연속학습으로 가는 길을 제시.
- 인용구: "This work paves the way for non-parametric continual learning for LLMs" [Q✓]; "achieving a 7% improvement in associative memory tasks over the state-of-the-art embedding model" [Q✓] — [arXiv abs](https://arxiv.org/abs/2502.14802)
- 검증: ✅ arXiv abs 확인(제목·저자 5인·v1 2025-02-20·초록) / venue 출처: arXiv Comments "ICML 2025. Code and data are available at …" (PMLR·icml.cc는 이번 세션에서 접속 불가로 재확인 못함)

**3. ★ A-MEM: Agentic Memory for LLM Agents** — Wujiang Xu et al. (Liang, Mei, Gao, Tan, Zhang; Rutgers 등)
- arXiv: [2502.12110](https://arxiv.org/abs/2502.12110) · 최초 공개: 2025-02-17 (v11: 2025-10-08)
- 베뉴: **NeurIPS 2025** — arXiv Comments "Advances in Neural Information Processing Systems (NeurIPS 2025)"; (초판 인용, 재접속 불가) [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/19909c36f51abc4856b4560aff3d36d6-Abstract-Conference.html), [poster 119020](https://neurips.cc/virtual/2025/poster/119020)
- 요약(KR):
  1. 기존 에이전트 메모리는 기본 저장·검색만 지원하고, 그래프 DB를 도입한 시도에도 정교한 메모리 조직이 부족하며 고정된 연산·구조 탓에 적응성이 낮다. 이 문제에 대해 Zettelkasten 원리에 따라 새 메모리마다 맥락 설명·키워드·태그 등 구조화 속성을 가진 노트를 만들고, 과거 메모리와 의미 있는 유사성이 있으면 링크해 상호연결된 지식 네트워크를 만드는 에이전틱 메모리를 제안.
  2. 새 메모리가 통합되면 기존 메모리의 맥락 표현·속성을 갱신하는 "메모리 진화"를 수행하며, 6개 파운데이션 모델 실험에서 기존 SOTA 베이스라인 대비 우수한 개선을 보고.
- 인용구: "create interconnected knowledge networks through dynamic indexing and linking" [Q✓] — [arXiv abs](https://arxiv.org/abs/2502.12110)
- 참고: Paper Digest가 "Most Influential NeurIPS 2025 Papers" 9위로 집계(2차 출처, 재확인 안 함) — [Paper Digest](https://www.paperdigest.org/certificate/bcf73fa6d42f9428)
- 검증: ✅ arXiv abs 확인(arXiv 표기 제목은 "A-MEM", 저자 6인, v1 2025-02-17, 초록) / venue 출처: arXiv Comments(NeurIPS 2025; 트랙 구분은 Comments에 없으며 초판의 "Main Conference Track"은 proceedings URL의 "Conference" 표기 기준)

**4. When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation (GraphRAG-Bench)** — Zhishang Xiang et al. (Wu, Zhang, Chen, Hong, Huang, Su)
- arXiv: [2506.05690](https://arxiv.org/abs/2506.05690) · 최초 공개: 2025-06-06 (v2: 2025-10-07, v3: 2026-02-22)
- 베뉴: **ICLR 2026** — arXiv v3 PDF 머리말 "Published as a conference paper at ICLR 2026"; 공식 repo News "[2026-01-26] Our GraphRAG Benchmark is accepted by ICLR'26" — [arXiv PDF](https://arxiv.org/pdf/2506.05690), [GitHub](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)
- 요약(KR):
  1. GraphRAG가 실제 과제 다수에서 vanilla RAG보다 자주 뒤처진다는 최근 보고에서 출발해, 계층적 지식 검색과 깊은 맥락 추론을 평가하는 GraphRAG-Bench를 제안. 사실 검색 → 복잡 추론 → 맥락 요약 → 창의적 생성으로 난이도가 높아지는 과제와, 그래프 구축·지식 검색·최종 생성 전 파이프라인에 대한 체계적 평가를 포함한다.
  2. 이 벤치마크로 GraphRAG가 기존 RAG를 능가하는 조건과 그 이유를 체계적으로 조사해 실무 적용 지침을 제시. "단순 사실 검색은 vanilla RAG가 동등/우세"라는 구체적 결론은 초록이 아닌 본문·GitHub에 있다.
- 인용구: "evaluate GraphRAG models on both hierarchical knowledge retrieval and deep contextual reasoning" [Q✓, arXiv 원문은 "onboth"로 붙어 표기] — [arXiv abs](https://arxiv.org/abs/2506.05690)
- 검증: ✅ arXiv abs 확인(제목·저자 7인·v1 2025-06-06·초록; arXiv Comments에는 베뉴 없음) / venue 출처: arXiv v3 PDF 머리말(ICLR 2026) + 저자 repo. OpenReview·proceedings.iclr.cc는 접속 불가로 직접 확인 못함

**5. LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval** — Yaoze Zhang et al. (Wu, Cai, Wang, Yan, Mao, Wang, Shi; Zhang·Wu 공동 1저자)
- arXiv: [2508.10391](https://arxiv.org/abs/2508.10391) · 최초 공개: 2025-08-14 (v4: 2025-11-12)
- 베뉴: **AAAI 2026** — arXiv Comments "Accepted by AAAI-26"; (초판 인용, 재접속 불가) [AAAI OJS vol. 40, pp. 34862–34869](https://ojs.aaai.org/index.php/AAAI/article/view/40789), [GitHub](https://github.com/razzzyz/leanrag)
- 요약(KR):
  1. 계층형 KG-RAG에서 상위 개념 요약들이 커뮤니티 간 추론에 필요한 명시적 관계가 없는 "의미의 섬(semantic islands)"으로 남고, 검색도 구조를 활용하지 못하는 평면 검색으로 퇴화한다는 두 문제를 지적. 엔티티 클러스터를 만들고 집계 요약들 사이에 명시적 관계를 새로 구성해 탐색 가능한 의미 네트워크를 만드는 의미 집계 알고리즘과, 질의를 세밀한 엔티티에 고정한 뒤 의미 경로를 따라 올라가는 상향식 구조 유도 검색을 제안.
  2. 그래프 경로 검색의 오버헤드와 중복 정보 검색을 줄이고, 도메인이 다른 4개 QA 벤치마크에서 응답 품질을 크게 개선하면서 검색 중복을 46% 줄였다고 보고.
- 인용구: "reducing 46\% retrieval redundancy" [Q✓] — [arXiv abs](https://arxiv.org/abs/2508.10391)
- 검증: ✅ arXiv abs 확인(제목·저자 8인·v1 2025-08-14·초록) / venue 출처: arXiv Comments "Accepted by AAAI-26" (AAAI OJS 권·쪽수는 접속 불가로 재확인 못함)

**6. ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning** — **Juyuan Wang** et al. (Zhao, Wei, Wang, Yu, Zhou, Xu, Xu) (소속 SCUT, Tencent WeChat AI, Pazhou Lab은 초판 기재값, 미대조)
- arXiv: [2508.10419](https://arxiv.org/abs/2508.10419) · 최초 공개: 2025-08-14 (v3: 2025-11-12)
- 베뉴: **AAAI 2026** — arXiv Comments "Accepted by AAAI 2026"; (초판 인용, 재접속 불가) [AAAI OJS vol. 40, pp. 33557–33565](https://ojs.aaai.org/index.php/AAAI/article/view/40644)
- 요약(KR):
  1. 장편 서사 이해에서 전통 RAG는 무상태(stateless)·단일 단계 검색 때문에 장거리 맥락 속 상호연결된 관계를 놓친다. 이에 대해 서사 추론을 새 증거 획득과 과거 지식 통합의 동적 상호작용으로 보는 ComoRAG를 제안. 추론이 막히면 동적 메모리 워크스페이스와 상호작용하는 반복 추론 사이클을 돌며, 탐색 질의를 생성하고 새로 검색한 증거를 전역 메모리 풀에 통합한다.
  2. 4개 장문 서사 벤치마크(200K+ 토큰)에서 가장 강한 베이스라인 대비 최대 11%의 일관된 상대 향상을 보고했고, 전역 맥락 이해가 필요한 복잡 질의에서 특히 유리했다.
- 인용구: "consistent relative gains up to 11% compared to the strongest baseline" [Q✓] — [arXiv abs](https://arxiv.org/abs/2508.10419)
- 검증: ✅ arXiv abs 확인(1저자 Juyuan Wang 확인, 저자 8인, v1 2025-08-14, 초록) / venue 출처: arXiv Comments "Accepted by AAAI 2026"

**7. GraphRAG-R1: Graph Retrieval-Augmented Generation with Process-Constrained Reinforcement Learning** — Chuanyue Yu et al. (Zhao, Li, Chang, Feng, Jiang, Sun, Li, Zhang, Li, Ziwei Zhang; 11인)
- arXiv: [2507.23581](https://arxiv.org/abs/2507.23581) · 최초 공개: 2025-07-31 (v2: 2026-01-27)
- 베뉴: **WWW 2026** — arXiv Comments "Accepted by the Web Conference 2026" — [arXiv abs](https://arxiv.org/abs/2507.23581). ACM DOI 10.1145/3774904.3792589는 HF README 기재값이며 doi.org 접속 불가로 미확인 — [HF README](https://huggingface.co/yuchuanyue/GraphRAG-R1/blob/main/README.md)
- 요약(KR):
  1. 기존 GraphRAG의 질의·검색 단계가 사전 정의 휴리스틱에 의존해 다중 홉 추론에서 병목을 겪는 문제에 대해, 과정 제약(process-constrained) 결과 기반 RL로 LLM을 훈련해 문제를 분해하고 검색 도구를 스스로 호출하게 하는 적응형 GraphRAG를 제안. rollout-with-thinking을 지원하는 수정 GRPO를 쓰고, 얕은 검색을 막는 PRA(Progressive Retrieval Attenuation) 보상과 과잉 사고를 비용과 균형 맞추는 CAF(Cost-Aware F1) 보상을 둔다. 콜드 스타트와 두 보상에 대응하는 3단계 phase-dependent 훈련과 그래프-텍스트 하이브리드 검색도 결합한다.
  2. 도메인 내·외 데이터셋 모두에서 SOTA GraphRAG 대비 복잡 추론 성능을 높였고, 다양한 기존 검색 방법과 유연하게 결합해도 일관된 개선을 보였다고 보고.
- 인용구: "adopts a hybrid graph-textual retrieval" [Q✓] — [arXiv abs](https://arxiv.org/abs/2507.23581)
- 주의: RL 검색 에이전트 일반은 타 연구자 범위이므로 "그래프 특화" 사례로만 포함.
- 검증: ✅ arXiv abs 확인(제목·저자 11인·v1 2025-07-31·초록) / venue 출처: arXiv Comments "Accepted by the Web Conference 2026"

**8. LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora** — Luyao Zhuang et al. (Chen, Xiao, Zhou, Zhang, Chen, Zhang, Huang; PolyU)
- arXiv: [2510.10114](https://arxiv.org/abs/2510.10114) · 최초 공개: 2025-10-11 (v4: 2025-11-09)
- 베뉴: **ICLR 2026** ⚠️(간접 확인) — 공저자(Qinggang Zhang·Xiao Huang) 그룹의 GraphRAG-Benchmark repo News "[2026-01-26] Our LinearRAG … is accepted by ICLR'26" — [GitHub](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark); 검색 색인상 [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10007603), OpenReview forum `mCtfkypdm6`, [PolyU 기관 리포지토리](https://ira.lib.polyu.edu.hk/handle/10397/119733) 존재(직접 접속 불가)
- 요약(KR):
  1. 기존 GraphRAG가 불안정하고 비싼 관계 추출에 의존해 잘못되거나 일관성 없는 관계가 담긴 노이즈 그래프를 만든다는 문제에 대해, 경량 엔티티 추출과 의미적 연결만으로 관계 없는(relation-free) 계층 그래프 "Tri-Graph"를 구축하는 LinearRAG를 제안. 그래프 구축 비용이 코퍼스 크기에 선형으로 늘고 추가 토큰을 소비하지 않는다.
  2. 검색은 (i) 국소 의미 브리징을 통한 관련 엔티티 활성화와 (ii) 전역 중요도 집계를 통한 패시지 검색의 2단계로 이루어지며, 4개 데이터셋에서 베이스라인을 크게 앞섰다고 보고.
- 인용구: "constructs a relation-free hierarchical graph, termed Tri-Graph" [Q✓] — [arXiv abs](https://arxiv.org/abs/2510.10114); "reduces indexing time by over 77%" [Q✓, 초록이 아닌 v4 PDF 본문(서론)] — [arXiv PDF](https://arxiv.org/pdf/2510.10114)
- 검증: ✅ arXiv abs 확인(제목·저자 8인·v1 2025-10-11·초록; Comments 없음, v4 PDF에 학회 머리말 없음) / venue 출처: ⚠️ 공저자 그룹 GitHub + 검색 색인. OpenReview·iclr.cc 접속 불가로 1차 확인 못함

**9. Does Memory Need Graphs? A Unified Framework and Empirical Analysis for Long-Term Dialog Memory** — Sen Hu et al. (Wei, Ran, Yao, Han, Wang, Chen, Lei Zou; PKU)
- arXiv: [2601.01280](https://arxiv.org/abs/2601.01280) · 최초 공개: 2026-01-03 (v3: 2026-02-10)
- 베뉴: **ACL 2026 Main** (Long Papers, 2026.acl-long.1232, pp. 26758–26782) — [ACL Anthology](https://aclanthology.org/2026.acl-long.1232/)
- 요약(KR):
  1. 그래프 구조가 대화 메모리 시스템에 점점 많이 쓰이지만 효과에 대한 실증 결과가 엇갈린다는 문제의식에서, 대화 메모리 시스템을 핵심 구성요소로 분해해 그래프·비그래프 방식을 모두 담는 통합 프레임워크를 만들었다. 이 위에서 LongMemEval·HaluMem에 대해 메모리 표현·조직·유지·검색의 설계 선택을 단계별 통제 실험으로 비교.
  2. 성능 차이의 상당 부분이 특정 아키텍처 혁신보다 기초 시스템 설정에서 비롯됨을 보이고, 향후 대화 메모리 연구를 위한 안정적이고 신뢰할 만한 강한 베이스라인을 제시.
- 인용구: "many performance differences are driven by foundational system settings rather than specific architectural innovations" [Q✓] — [arXiv abs](https://arxiv.org/abs/2601.01280)
- 검증: ✅ arXiv abs 확인(제목·저자 8인·v1 2026-01-03·초록; Comments 없음) / venue 출처: ACL Anthology 2026.acl-long.1232 ("Proceedings of the 64th Annual Meeting of the ACL (Volume 1: Long Papers)"). Anthology의 저자 순서는 Han이 Yao보다 앞이다(arXiv는 Yao→Han).

**10. MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents** — Dongming Jiang et al. (Yi Li, Guanpeng Li, Bingzhe Li)
- arXiv: [2601.03236](https://arxiv.org/abs/2601.03236) · 최초 공개: 2026-01-06 (v2: 2026-04-16)
- 베뉴: **ACL 2026 Main** (Long, 2026.acl-long.1709, pp. 36848–36865) — [ACL Anthology](https://aclanthology.org/2026.acl-long.1709/)
- 요약(KR):
  1. 기존 메모리 증강 생성(MAG)이 단일(monolithic) 메모리 저장소의 의미 유사도에 의존해 시간·인과·엔티티 정보가 뒤섞이고, 해석 가능성과 질의 의도-증거 정렬이 떨어진다는 문제에 대해, 각 메모리 항목을 의미·시간·인과·엔티티의 직교 그래프에 걸쳐 표현하고 검색을 이 관계 뷰들 위의 정책 유도 순회로 정식화.
  2. 메모리 표현과 검색 로직을 분리해 질의 적응적 뷰 선택, 투명한 추론 경로, 세밀한 검색 제어를 제공하며, LoCoMo·LongMemEval의 장기 추론 과제에서 최신 에이전틱 메모리 시스템을 일관되게 능가했다고 보고.
- 인용구: "formulates retrieval as policy-guided traversal over these relational views" [Q✓] — [arXiv abs](https://arxiv.org/abs/2601.03236), [ACL Anthology](https://aclanthology.org/2026.acl-long.1709/)
- 검증: ✅ arXiv abs 확인(제목·저자 4인·v1 2026-01-06·초록) / venue 출처: arXiv Comments "ACL 2026 Main" + ACL Anthology 2026.acl-long.1709

**11. STEM: Structure-Tracing Evidence Mining for Knowledge Graphs-Driven Retrieval-Augmented Generation** — Peng Yu et al. (En Xu, Bin Chen, Haibiao Chen, Yinfei Xu)
- arXiv: [2604.22282](https://arxiv.org/abs/2604.22282) · 최초 공개: 2026-04-24 (v2: 2026-05-18)
- 베뉴: **ACL 2026 Main** (Long, 2026.acl-long.329, pp. 7252–7285; arXiv Comments에 "Main Conference, Oral Presentation") — [ACL Anthology](https://aclanthology.org/2026.acl-long.329/)
- 요약(KR):
  1. KG 기반 QA에서 KG의 구조적 이질성이 검색 시 의미 불일치를 낳고, 기존 추론 경로 검색은 전역 구조 관점이 없다는 두 문제에 대해, 다중 홉 추론을 스키마 유도 그래프 탐색으로 재정식화한 STEM을 제안. Semantic-to-Structural Projection으로 KG 구조 사전지식을 이용해 질의를 원자적 관계 단언으로 분해하고 적응형 질의 스키마 그래프를 만든 뒤, 전역 인식 노드 앵커링·서브그래프 검색으로 증거 추론 그래프를 얻는다. Triple-Dependent GNN(Triple-GNN)이 구축을 이끄는 Global Guidance Subgraph를 생성한다.
  2. 다중 홉 추론 그래프 검색의 정확도와 증거 완전성을 크게 높여 여러 다중 홉 벤치마크에서 SOTA를 보고.
- 인용구: "reframes multi-hop reasoning as a schema-guided graph search task" [Q✓] — [arXiv abs](https://arxiv.org/abs/2604.22282)
- 검증: ✅ arXiv abs 확인(제목·저자 5인·v1 2026-04-24·초록) / venue 출처: arXiv Comments "accepted to ACL 2026 (Main Conference, Oral Presentation)" + ACL Anthology 2026.acl-long.329

**12. AtomicRAG: Atom-Entity Graphs for Retrieval-Augmented Generation** — Yanning Hou et al. (Yuan, Zhou, Chen, Liang, Wang, Liu, Huang; 총 8인)
- arXiv: [2604.20844](https://arxiv.org/abs/2604.20844) · 최초 공개: 2026-02-10 (arXiv Submission history의 v1 날짜. ID는 2604이고 PDF 머리말은 "Preprint. April 24, 2026"이라 공개(announce)는 2026-04로 보임)
- 베뉴: **Preprint (arXiv)** — Comments 없음, PDF 머리말 "Preprint". 이름이 비슷한 **Atom-RAG**(Seok Jun Hwang et al., EMNLP 2026 Findings, [2026.findings-emnlp.2323](https://aclanthology.org/2026.findings-emnlp.2323/))는 별개 논문이니 혼동 주의
- 요약(KR):
  1. 텍스트 청크를 지식 표현의 기본 단위로 쓰면 여러 원자적 사실이 경직되게 묶이고, 트리플 기반 엔티티 연결은 관계 추출 오류에 민감해 추론 경로가 누락·오류된다는 문제에 대해, 지식을 자기완결적 사실 단위인 "지식 원자(knowledge atom)"로 저장하고 엔티티 간 엣지는 관계의 존재 여부만 나타내는 Atom-Entity Graph를 제안.
  2. Personalized PageRank와 관련성 기반 필터링을 결합해 정확한 엔티티 연결과 추론 신뢰성을 유지하며, 이론 분석과 5개 공개 벤치마크 실험에서 강한 RAG 베이스라인 대비 검색 정확도·추론 견고성 우위를 보고.
- 인용구: "rigidly groups multiple atomic facts together" [Q✓] — [arXiv abs](https://arxiv.org/abs/2604.20844), [GitHub](https://github.com/7HHHHH/AtomicRAG)
- 검증: ✅ arXiv abs 확인(제목·저자 8인·v1 2026-02-10·초록; 제목 하이픈은 "Atom-Entity") / venue 출처: arXiv Comments 없음 + PDF "Preprint" + 검색에서 채택 정보 없음 → Preprint (arXiv)

**13. Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation (Ψ-RAG)** — Ziwen Zhao, Menglin Yang
- arXiv: [2605.00529](https://arxiv.org/abs/2605.00529) · 최초 공개: 2026-05-01
- 베뉴: **ICML 2026** — arXiv Comments "ICML 2026"; arXiv PDF 각주 "Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026"; GitHub "May 1, 2026: Our paper is accepted in ICML'26" — [GitHub](https://github.com/Newiz430/Psi-RAG); (재접속 불가) [ICML 2026 poster 66483](https://icml.cc/virtual/2026/poster/66483)
- 요약(KR):
  1. 단일 문서용 Tree-RAG를 문서 간 다중 홉 질의로 확장할 때 생기는 세 문제, 즉 (1) k-means의 경직된 분포 가정에서 오는 노이즈, (2) 트리 인덱스 간 명시적 교차 문서 연결 부재(구조적 고립), (3) 세부를 가리는 거친 추상화에 대해, 사전 분포 가정 없이 반복적 "병합·붕괴(merging and collapse)"로 만드는 계층적 추상 트리 인덱스와, 질의 재구성과 에이전트 기반 하이브리드 검색기로 지식베이스와 상호작용하는 다중 입도 검색 에이전트를 결합한 Ψ-RAG를 제안.
  2. 토큰 수준 QA부터 문서 수준 요약까지 지원하며, 문서 간 다중 홉 QA에서 평균 F1 기준 RAPTOR 대비 25.9%, HippoRAG 2 대비 7.4% 향상을 보고.
- 인용구: 'an iterative "merging and collapse" process' [Q✓]; "outperforms RAPTOR by 25.9% and HippoRAG 2 by 7.4% in average F1 score" [Q✓] — [arXiv abs](https://arxiv.org/abs/2605.00529)
- 검증: ✅ arXiv abs 확인(제목·저자 2인·v1 2026-05-01·초록) / venue 출처: arXiv Comments "ICML 2026" + arXiv PDF PMLR 306 각주

#### 보조 후보 (범위 내이나 본 목록에서 제외/보완용)
- **HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation** (Haoran Luo et al., [2503.21322](https://arxiv.org/abs/2503.21322), v1 2025-03-27) — **NeurIPS 2025** — 일반 그래프 엣지는 두 엔티티만 이어 n-ary 관계(n ≥ 2)를 표현하지 못하므로 하이퍼엣지로 n-ary 관계 사실을 표현. 의료·농업·CS·법률에서 표준 RAG 및 기존 그래프 RAG 대비 정확도·검색 효율·생성 품질 개선을 보고 — [arXiv abs](https://arxiv.org/abs/2503.21322)
  - 검증: ✅ arXiv abs 확인 / venue 출처: arXiv Comments "Accepted by NeurIPS 2025 main conference" + Journal-ref "NeurIPS 2025"
- **RAG vs. GraphRAG: A Systematic Evaluation and Key Insights** (Haoyu Han et al., [2502.11371](https://arxiv.org/abs/2502.11371), v1 2025-02-17; v3 2026-03-04; **v4 2026-11-07**) — **KDD 2026** ⚠️ — 텍스트 벤치마크(QA, 질의 기반 요약)에서 데이터 전처리·검색 구성·생성 설정을 표준화한 통합 평가 프로토콜로 RAG와 GraphRAG를 비교. 과제·평가 관점에 따라 강점이 다름을 보이고, 둘을 선택·통합하는 전략으로 일관된 개선을 얻었으며, 실패 유형·효율 트레이드오프·평가 편향을 분석 — [arXiv abs](https://arxiv.org/abs/2502.11371), [arXiv v4 PDF](https://arxiv.org/pdf/2502.11371)
  - 검증: ⚠️ arXiv abs 확인(Comments 없음) / venue 출처: v4 PDF의 ACM 저작권 블록과 참고 형식 "Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '26), August 09–13, 2026, Jeju", DOI 10.1145/3770855.3817575, ACM ISBN 979-8-4007-2259-2. 실제 DOI가 박힌 camera-ready 형식이라 채택 가능성이 높지만, doi.org·ACM DL 접속 불가로 1차 확인은 못함 (초판의 "v3 템플릿 placeholder" 판단은 v4에서 바뀜)
- **Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning** (Haoran Luo et al., [2507.21892](https://arxiv.org/abs/2507.21892), v1 2025-07-29; v2 2026-06-02) — **ICML 2026** — 경량 지식 하이퍼그래프 구축, 다중 턴 에이전트-환경 상호작용으로서의 검색, 엔드투엔드 보상으로 최적화하는 "최초의 에이전틱 GraphRAG"를 표방. 기존 GraphRAG 및 RL 강화 RAG 대비 추론 정확도·검색 효율·생성 품질 개선을 보고 — [arXiv abs](https://arxiv.org/abs/2507.21892)
  - 검증: ✅ arXiv abs 확인 / venue 출처: arXiv Comments "Accepted by ICML 2026 main conference" + Journal-ref "ICML 2026" + v2 PDF 각주(PMLR 306, 2026). 초판의 "Unverified"를 정정
- **Deep GraphRAG: A Balanced Approach to Hierarchical Retrieval and Adaptive Integration** (Yuejie Li et al., [2601.11144](https://arxiv.org/abs/2601.11144), v1 2026-01-16; v3 2026-01-29) — **Preprint (arXiv)** — 커뮤니티 간 필터링 → 커뮤니티 수준 정제 → 커뮤니티 내 엔티티 수준 세밀 검색의 3단계 전역→국소 계층 검색, 빔서치 최적화 동적 리랭킹, DW-GRPO(Dynamic Weighting Reward GRPO; 관련성·충실성·간결성 보상 가중 동적 조정)로 학습한 소형 LLM 지식 통합 모듈. 1.5B 모델이 통합 과제에서 70B 성능에 근접하며 NQ·HotpotQA에서 평가 — [arXiv abs](https://arxiv.org/abs/2601.11144)
  - 검증: ✅ arXiv abs 확인(Comments 없음 → Preprint). 초판의 "Ant Group" 소속은 미대조
- **ZoomRAG: Hierarchical Random-walk Zooming across Multi-scale Information Graphs for Fast and Accurate RAG** (Xianming Hu et al.) — **ACL 2026 Findings** (2026.findings-acl.1643, pp. 32840–32859) — 전역 관계 그래프에서 질의 기반 랜덤워크로 소수 문서를 찾고, 선택 문서 안으로 "줌인"해 세밀한 의미·시간 관계 그래프에서 두 번째 랜덤워크로 증거 청크를 찾는 coarse-to-fine 검색, 병렬 처리 포함 — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1643/)
  - 검증: ✅ ACL Anthology 확인(Findings of the ACL: ACL 2026) / arXiv ID는 찾지 않음(Anthology 엔트리로 존재 확인)
- **Retrieval-Augmented Generation with Retrieval-optimized TRA Structure** (JungHyoun Kim, Soohyeong Kim, Seok Jun Hwang, Jeonghyeon Park, Yong Suk Choi) — **EMNLP 2026 Main** (2026.emnlp-main.118, pp. 2735–2762) — LLM으로 코퍼스를 트리플 그래프로 바꾸면 계층 정보를 표현하지 못하고 구조가 경직돼 RAG에 부적합할 수 있다고 지적. Title-Relations-Attributes(TRA) 데이터베이스 구조를 제안하고, 질의 분해와 결합했을 때 강한 다중 홉 추론 성능을 보고 — [ACL Anthology](https://aclanthology.org/2026.emnlp-main.118/)
  - 검증: ✅ ACL Anthology 확인(Proceedings of EMNLP 2026)
- **Beyond Dialogue Time: Temporal Semantic Memory for Personalized LLM Agents** (Miao Su et al.) — **ACL 2026 Findings** (2026.findings-acl.1496, pp. 29935–29951) — 대화 시점이 아닌 의미적 시간(실제 발생 시점) 타임라인으로 메모리를 조직하고 지속(durative) 메모리를 구축·활용 — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1496/)
  - 검증: ✅ ACL Anthology 확인. 정식 제목에 "Beyond Dialogue Time:" 부제가 붙어 있어 정정
- 기타 2026 그래프 RAG 논문(모두 arXiv abs 확인):
  - GraFine (Seonho An et al., [2601.18579](https://arxiv.org/abs/2601.18579), v1 2026-01-26): 코퍼스 그래프 위 검색 시점 정제(SPX 의미 인지 추가 + GSR 그래프 인지 가지치기). **CIKM 2026** (arXiv Comments "CIKM 2026")
  - AGRAG (Yubo Wang et al., [2511.05549](https://arxiv.org/abs/2511.05549), v1 2025-11-02): LLM 엔티티 추출을 **통계 기반 방법**으로 대체해 환각·오류 전파를 막고(초판의 "TF-IDF" 표현은 초록에 없음), 추론을 MCMI(Minimum Cost Maximum Influence) 서브그래프 생성 문제로 정식화. **ICDE 2026** (arXiv Comments "ICDE 2026 Camera-ready")
  - EraRAG (Fangyuan Zhang et al., [2506.20963](https://arxiv.org/abs/2506.20963), v1 2025-06-26): 초평면 LSH로 다층 그래프를 구성해 성장 코퍼스에 국소적 증분 삽입을 지원. 업데이트 시간·토큰 최대 10배 감소 보고. **Preprint** (arXiv Comments "Under review")
  - LivingRAG: Augmenting Graph RAG with Experience (Yuzhuo Cui et al., [2608.25960](https://arxiv.org/abs/2608.25960), v1 2026-08-26): 그래프 RAG에 쓰기 가능한 검증된 추론 경험 저장소를 추가. **Preprint**
  - SVD-RAG (Zhihui Sun, 단독 저자, [2607.10316](https://arxiv.org/abs/2607.10316), v1 2026-07-11): RAPTOR 트리의 LLM 추상 요약을 문장 임베딩 행렬 SVD 기반 추출 요약으로 대체해 토큰 약 85% 절감, 트리 구축 317배 가속. **Preprint**
- ~~Token-Free Hierarchical Indexing for RAG beyond LLM-based Summarization (SeRAG)~~ — **목록에서 제거**: 초판의 유일한 근거였던 icml.cc poster 63668에 접속할 수 없었고, arXiv 제목 검색("Token-Free Hierarchical Indexing", "beyond LLM-based Summarization")과 웹 검색에서도 찾지 못함. OpenReview는 접속 불가 → Unverified로 제외

### Inferences
- 검증된 베뉴 기준 채택 흐름: EMNLP 2025 Findings(LightRAG) · ICML 2025(HippoRAG 2) → NeurIPS 2025(HyperGraphRAG, A-MEM) → AAAI 2026(LeanRAG, ComoRAG) → ICLR 2026(GraphRAG-Bench; LinearRAG ⚠️) → WWW 2026(GraphRAG-R1) · ICDE 2026(AGRAG) → ACL 2026 Main(MAGMA, STEM, Does Memory Need Graphs?, MegaRAG, APEX-MEM) / ACL 2026 Findings(ZoomRAG, TSM, TiMem) → ICML 2026(Ψ-RAG, Graph-R1) → KDD 2026(RAG vs. GraphRAG ⚠️) · CIKM 2026(GraFine) → EMNLP 2026 Main(TRA). 그래프/구조 RAG는 ML·NLP·웹/IR·DB 베뉴 전반에서 꾸준히 채택되는 주류 하위 분야로 볼 수 있다.
- HippoRAG 2는 AtomicRAG의 PPR 기반 검색과 Ψ-RAG의 비교 기준선(초록에 HippoRAG 2 대비 +7.4% 명시)으로 계속 등장한다. 2026년의 "사실상 강한 그래프 베이스라인"으로 기능하는 셈이다. 초판의 "LinearRAG PPR" 언급은 초록 근거가 없어 제외했고, LinearRAG 초록은 "global importance aggregation"으로만 기술한다.
- LightRAG의 영향력은 학회 등급(Findings)보다 오픈소스 채택에서 비롯된 것으로 보인다(추정, 정량 근거 미수집).

### Gaps
- OpenReview(api2/openreview.net), iclr.cc, icml.cc, neurips.cc, PMLR, AAAI OJS, doi.org가 이번 세션 네트워크에서 차단되어 있었다. 그래서 ICLR/ICML/NeurIPS/AAAI 베뉴는 arXiv Comments·PDF 머리말로만 확인했고, OpenReview decision 라벨(Poster/Spotlight/Oral)은 확인하지 못했다.
- LinearRAG의 ICLR 2026 채택은 1차 페이지(OpenReview/iclr.cc)로 확인하지 못함(공저자 그룹 GitHub + 검색 색인뿐).
- RAG vs. GraphRAG의 KDD 2026은 arXiv v4 PDF의 ACM 블록(DOI 포함)에만 근거하며, ACM DL 확인은 못함. GraphRAG-R1의 ACM DOI도 미확인.
- 초판 기재 PMLR/AAAI OJS 권·쪽수(HippoRAG 2 pp. 21497–21515, LeanRAG pp. 34862–34869, ComoRAG pp. 33557–33565)는 이번에 재확인하지 못함(접속 불가).
- NodeRAG(2504.11544), PathRAG(2502.14902), Mem0(2504.19413), Zep(2501.13956), GFM-RAG(2502.01113), ArchRAG(2502.09891)는 범위 확인을 하지 않아 목록에서 제외(누락이 중요도 판단을 의미하지 않음).

## Q2. 이 하위 분야는 어떻게 세분되는가?

### Takeaway
대표 서베이는 파이프라인 단계(그래프 인덱싱 / 그래프 유도 검색 / 그래프 강화 생성)로 분류하지만, 2025–26년 논문 흐름은 "무엇을 그래프로 만들고 어떻게 순회하는가"를 기준으로 6개 하위 범주로 나누는 편이 실용적이다: (A) KG 구축형 GraphRAG, (B) 경량/관계-free 그래프 인덱싱, (C) 계층 요약 트리·커뮤니티 계층, (D) 메모리-as-그래프(에이전트 장기기억), (E) 그래프+에이전트 추론(KG 추론/RL), (F) 실증 분석·벤치마크.

### Cited Findings
- 서베이 분류 1: "Graph Retrieval-Augmented Generation: A Survey"(Boci Peng et al., [2408.08921](https://arxiv.org/abs/2408.08921), v1 2024-08-15)는 Graph-Based Indexing(G-Indexing), Graph-Guided Retrieval(G-Retrieval), Graph-Enhanced Generation(G-Generation)의 3단계로 정리 — [arXiv abs](https://arxiv.org/abs/2408.08921). arXiv Comments는 "Ongoing work"이며, ACM TOIS 게재 여부는 확인 못함.
  - 검증: ✅ arXiv abs 확인(제목·1저자·v1·초록의 3단계 용어)
- 서베이 분류 2: "Retrieval-Augmented Generation with Graphs (GraphRAG)"(Haoyu Han et al., [2501.00309](https://arxiv.org/abs/2501.00309), v1 2024-12-31)는 query processor, retriever, organizer, generator, data source의 5요소 프레임워크로 정리 — [arXiv abs](https://arxiv.org/abs/2501.00309)
  - 검증: ✅ arXiv abs 확인(초록에 5요소 문구 그대로 존재), Comments 없음
- 서베이 분류 3: "Graph-based Approaches and Functionalities in Retrieval-Augmented Generation: A Comprehensive Survey"(Zulun Zhu, Tiancheng Huang et al., [2504.10499](https://arxiv.org/abs/2504.10499), v1 2025-04-08, v2 2026-01-21)는 database construction, algorithms, pipelines, tasks 관점의 분류를 제시 — [arXiv abs](https://arxiv.org/abs/2504.10499)
  - 검증: ✅ arXiv abs 확인, Comments 없음
- 에이전트 메모리 쪽: "Does Memory Need Graphs?"(ACL 2026 Main)는 대화 메모리 설계 선택을 **메모리 표현·조직·유지·검색(representation, organization, maintenance, retrieval)**으로 나눠 비교 — [arXiv abs](https://arxiv.org/abs/2601.01280). (초판의 "추출·인덱싱·검색·QA 4단계"는 초록 근거가 없어 정정)
- 서베이 "Graph-Based Personalized Memory for LLM Agents: Representation, Evolution, Retrieval, and Evaluation"(Dac Duy Anh Nguyen et al., [2609.08599](https://arxiv.org/abs/2609.08599), v1 2026-09-08)은 그래프 기반 개인화 메모리를 표현·진화·검색·평가의 생애주기 관점으로 정리 — [arXiv abs](https://arxiv.org/abs/2609.08599)
  - 검증: ✅ arXiv abs 확인 / venue: **ICKG 2026** (arXiv Comments "Accepted by ICKG 2026")

#### 제안 하위 범주와 논문 배치
| 하위 범주 | 핵심 아이디어 | 해당 논문 (본 목록 / 보조) |
|---|---|---|
| (A) KG/엔티티-관계 그래프 구축형 GraphRAG | LLM으로 엔티티·관계(또는 n-ary) 추출 → 그래프/KG 위에서 검색 | LightRAG★, STEM(KG-driven, 스키마 유도 그래프 탐색) / HyperGraphRAG, MegaRAG(멀티모달, ACL 2026 Main, 범위 외) |
| (B) 경량·관계-free·원자 단위 그래프 인덱싱 | 관계 추출 비용·노이즈 제거, PPR/중요도 집계 기반 확산 | HippoRAG 2★, LinearRAG, AtomicRAG / AGRAG(ICDE 2026), GraFine(CIKM 2026) |
| (C) 계층 요약 트리·커뮤니티 계층(RAPTOR/MS GraphRAG 계열) | 군집·요약으로 다층 인덱스, 상향/하향 순회 | LeanRAG, Ψ-RAG / Deep GraphRAG, ZoomRAG, SVD-RAG, EraRAG |
| (C') 구조 인식 RAG(표·DB·문서 구조) | 데이터 고유 구조를 인덱스로 사용 | / TRA structure(EMNLP 2026 Main), Semantic Compression Trees(원문 계층 구조 보존·의미 잔차 노드) |
| (D) 메모리-as-그래프 / 메모리 증강 RAG | 에이전트·장문 상호작용 기억을 링크/다중 그래프로 저장, 검색=순회 | A-MEM★, MAGMA, ComoRAG, HippoRAG 2★(가교) / Temporal Semantic Memory(ACL 2026 Findings), TiMem(ACL 2026 Findings), APEX-MEM(ACL 2026 Main), MemORAI("ACL Findings") |
| (E) 그래프+에이전트 추론(KG 추론 에이전트, RL) | LLM이 그래프 탐색을 정책으로 학습/수행 | GraphRAG-R1, MAGMA(정책 유도 순회), Ψ-RAG(다중 입도 검색 에이전트) / Graph-R1(ICML 2026), Deep GraphRAG, HyperGraphPro |
| (F) 실증 분석·벤치마크 | "그래프가 언제 도움이 되는가", 통제 비교 | GraphRAG-Bench, Does Memory Need Graphs? / RAG vs. GraphRAG |

(근거: 각 논문 항목의 Q1 출처 참조. Semantic Compression Trees(Junaid Farooq, v1 2026-08-21, Preprint): 각 노드가 부모 대비 추가 정보인 "semantic residual"만 저장하고 루트에서 점진적으로 내려가며 검색 — [arXiv abs 2608.21610](https://arxiv.org/abs/2608.21610). TiMem: "Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents"(Kai Li et al., [2601.02845](https://arxiv.org/abs/2601.02845), arXiv Comments "ACL 2026 Findings"). APEX-MEM: "Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI"(Pratyay Banerjee et al., [2604.14362](https://arxiv.org/abs/2604.14362), arXiv Comments "Accepted to ACL 2026 Mains"[원문 그대로]). MemORAI(Hung Pham Van et al., [2605.01386](https://arxiv.org/abs/2605.01386), arXiv Comments "ACL Findings", 연도 미표기). 이 셋은 arXiv Comments로만 확인했고 ACL Anthology 엔트리는 미대조.)

### Inferences
- (B)와 (C)는 모두 "MS GraphRAG/LightRAG식 LLM 관계 추출은 비싸고 노이즈가 많다"는 공통 비판에서 출발하지만, 해법의 방향은 다르다(평평하고 싼 그래프 vs. 더 잘 연결된 계층). LinearRAG·AtomicRAG·AGRAG 초록이 모두 관계/엔티티 추출 오류를 명시적 동기로 든다.
- (D)와 (E)의 경계가 2026년에 흐려지는 중이다. MAGMA는 메모리 구조(D)이면서 검색을 정책 순회(E)로 정의하고, Ψ-RAG는 트리 인덱스(C)에 검색 에이전트(E)를 결합한다.
- (F)는 별도 범주로 둘 가치가 있다. 2025–26년 실증 연구들이 (A)–(D) 각 범주의 가치 주장을 직접 반박하거나 조건부로 만들고 있기 때문이다.

### Gaps
- 위 6범주 분류는 서베이의 공식 분류가 아니라 수집 논문을 바탕으로 한 제안이다. 2026년 최신 GraphRAG 서베이의 분류 체계는 1차 확인하지 못했다.
- 구조 인식 RAG(표·문서 구조) 분야는 대표 논문을 충분히 확보하지 못했다(TRA는 Anthology 초록만 확인, 본문 미확인).

## Q3. 비판적·실증적 연구는 GraphRAG가 언제 vanilla RAG를 이기며, 비용은 어떻다고 말하는가?

### Takeaway
실증 연구들의 공통 결론은 "그래프는 조건부로만 이득"이라는 것이다. 단일 홉·국소 사실 검색에서는 vanilla RAG가 동등하거나 우세하고, 다중 홉 추론·전역 요약·연상 기억에서 그래프가 이득을 보인다. 에이전트 메모리에서도 성능 차이의 상당 부분이 그래프 구조 자체보다 기초 설정에서 온다는 결과가 나왔다. 비용 측면에서는 LLM 기반 그래프 구축과 긴 프롬프트의 "토큰 세금"이 핵심 약점이며, 2025–26년 방법들(LinearRAG, LeanRAG, EraRAG, SVD-RAG)은 이를 줄이는 방향으로 움직이고 있다.

### Cited Findings
- GraphRAG-Bench(ICLR 2026)는 GraphRAG가 실제 과제에서 vanilla RAG보다 자주 뒤처진다는 최근 보고를 동기로 삼음 — [arXiv abs](https://arxiv.org/abs/2506.05690) [✅ 초록 확인]
- GraphRAG-Bench 서론이 인용한 선행 결과: "the previous study (Han et al., 2025) demonstrates that GraphRAG achieves 13.4% lower" accuracy(NQ, vanilla RAG 대비). GraphRAG-Bench 자체 실험 수치가 아니다 — [arXiv PDF v3](https://arxiv.org/pdf/2506.05690) [✅ PDF 본문 확인]
- GraphRAG-Bench 결론: 단순 사실 검색에서는 기본 RAG가 동등/우세하고 그래프가 중복·노이즈를 주입할 수 있으며, 복잡한 다중 홉 추론에서 그래프의 이점이 나타남 — [GitHub](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) (⚠️ 초록에는 "조건과 이유를 조사"까지만 있고 이 구체 결론은 본문/README 기준; 본문 결론 절은 대조하지 않음)
- 2차 출처 수치(⚠️ 부분 확인): 복잡 추론 53.4 vs 42.9, 맥락 요약 64.4 vs 51.3(그래프 vs 텍스트 청크), 의료 요약에서 reranked RAG 65.75% vs HippoRAG2 63.08% — [papernotes 요약](https://en.papernotes.org/ICLR2026/information_retrieval/when_to_use_graphs_in_rag_a_comprehensive_analysis_for_graph_retrieval-augmented/), [VentureBeat](https://venturebeat.com/orchestration/stop-graphing-everything-when-graphrag-actually-beats-vector-rag). v3 PDF 텍스트에서 65.75, 63.08, 64.40, 51.30, 42.93은 표 값으로 존재함을 확인했으나 53.4는 찾지 못했고, 각 값의 행·열(방법·과제) 대응은 대조하지 못함
- RAG vs. GraphRAG(Han et al., KDD 2026 ⚠️): v4 초록은 "과제·평가 관점별로 RAG와 GraphRAG의 강점이 다르며, 둘을 선택·통합하는 전략이 일관된 개선을 준다"고 기술 — [arXiv abs](https://arxiv.org/abs/2502.11371). 초판의 "QA에서 RAG는 단일 홉, GraphRAG는 추론 중심 질문에 강함"은 v3 본문 기준이며 v4 초록에는 없다(v4 본문은 미대조)
- Does Memory Need Graphs?(ACL 2026 Main): 대화 메모리에서 성능 차이의 상당 부분이 특정 아키텍처 혁신보다 기초 시스템 설정에서 비롯됨 — [arXiv abs](https://arxiv.org/abs/2601.01280) [✅]
- 비용 개선 사례: LeanRAG는 검색 중복 46% 감소 — [arXiv abs](https://arxiv.org/abs/2508.10391) [✅]; LinearRAG는 그래프 구축이 코퍼스 크기에 선형이고 추가 토큰 소비가 없으며(초록), 본문 서론에서 "reduces indexing time by over 77%" — [arXiv abs](https://arxiv.org/abs/2510.10114), [arXiv PDF](https://arxiv.org/pdf/2510.10114) [✅]; EraRAG는 기존 Graph-RAG 대비 업데이트 시간·토큰 소비 최대 한 자릿수(10배) 감소 — [arXiv abs](https://arxiv.org/abs/2506.20963) [✅]; SVD-RAG는 RAPTOR식 LLM 요약 대비 토큰 약 85% 절감, 검색 품질은 RAPTOR의 1–5% 이내 — [arXiv abs](https://arxiv.org/abs/2607.10316) [✅]
- 트리(RAPTOR) 계열에 대한 TreeHop([2504.20114](https://arxiv.org/abs/2504.20114), "TreeHop: Efficient Embedding-Level Query Rewriter") 관련 정정: 초판의 "top-5 RAPTOR가 여러 벤치마크에서 직접 검색보다 뒤처진다"는 주장은 v1(2025-04-28)·v3(2026-08-10) PDF 어디에서도 찾지 못했다(v1에는 RAPTOR 언급 없음). ❌ 삭제. v3 본문은 오히려 RAPTOR가 HotpotQA 3회차 예산에서 TreeHop보다 recall이 높고 체인 recall도 55.1% vs 54.9%로 근소 우위이며, TreeHop의 이점은 RAPTOR 대비 지연 96.4–98.3% 감소라고 기술 — [arXiv PDF v3](https://arxiv.org/pdf/2504.20114)
- Ψ-RAG(ICML 2026)는 문서 간 다중 홉 QA에서 RAPTOR 대비 F1 +25.9%, HippoRAG 2 대비 +7.4%를 보고(트리 구조가 문서 간 연결을 갖추면 이득) — [arXiv abs](https://arxiv.org/abs/2605.00529) [✅]

### Inferences
- 실무 판단 규칙(추론): 질의가 국소 사실 조회 위주면 vanilla RAG(+리랭커)를, 다중 홉·전역 요약·장기 연상 기억이 핵심이면 그래프/계층 인덱스를 고려한다. GraphRAG-Bench, RAG vs. GraphRAG, Does Memory Need Graphs?에서 공통으로 도출되는 점은 그래프 도입 전에 "강한 평면 베이스라인"과의 통제 비교가 필수라는 것이다.
- 2026년 신규 방법들이 정확도뿐 아니라 "구축 비용·중복·토큰 감소"를 헤드라인 수치로 내세우는 것(LeanRAG 46%, LinearRAG 77%, SVD-RAG 85%)은 실증 연구들이 제기한 비용 비판에 대한 대응으로 해석할 수 있다.

### Gaps
- GraphRAG-Bench의 토큰 비용 수치("token tax")와 표 값의 행·열 의미를 원문 표로 대조하지 못함.
- "LightRAG·HippoRAG가 증거 재현율은 다소 높이지만 컨텍스트 관련도는 vanilla RAG보다 크게 낮다"는 결과는 검색 요약에 등장했으나 출처 논문을 특정하지 못해 인용에서 제외.
- "In-depth Analysis of Graph-based RAG in a Unified Framework"(Yingli Zhou et al., [2503.04338](https://arxiv.org/abs/2503.04338), v1 2025-03-06, v2 2026-04-27)와 "GraphRAG-Bench: Challenging Domain-Specific Reasoning for Evaluating Graph Retrieval-Augmented Generation"(Yilin Xiao et al., [2506.02404](https://arxiv.org/abs/2506.02404), v1 2025-06-03)은 arXiv에서 존재를 확인했으나 결과는 분석하지 않음. 이름이 같은 두 "GraphRAG-Bench"(2506.05690 Xiang et al. vs 2506.02404 Xiao et al.)가 있으니 보고서 작성 시 혼동에 주의.

## Q4. 2026년의 새로운 방향은?

### Takeaway
2026년의 신호는 여섯 가지다: (1) 관계 추출을 버리거나 단위를 바꾼 그래프(지식 원자, relation-free), (2) 저비용 계층 인덱싱과 문서 간 트리, (3) 다중 그래프(의미·시간·인과·엔티티) 에이전트 메모리와 시간적 메모리, (4) 그래프 검색을 정책으로 학습하는 에이전트화, (5) 연합/분산 GraphRAG, (6) 그래프 메모리 효과에 대한 회의적 실증 분석.

### Cited Findings
- 지식 원자 그래프: AtomicRAG(Preprint, v1 2026-02-10)는 청크 대신 자기완결적 지식 원자와 Atom-Entity Graph를 사용 — [arXiv abs](https://arxiv.org/abs/2604.20844) [✅]
- 관계-free 그래프: LinearRAG(ICLR 2026 ⚠️) — [arXiv abs](https://arxiv.org/abs/2510.10114)
- 저비용·문서 간 계층 인덱스: Ψ-RAG(ICML 2026) — [arXiv abs](https://arxiv.org/abs/2605.00529); ZoomRAG(ACL 2026 Findings) — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1643/); SVD-RAG(Preprint) — [arXiv abs](https://arxiv.org/abs/2607.10316). (초판의 SeRAG는 존재 미확인으로 제거)
- 다중 그래프 에이전트 메모리: MAGMA(ACL 2026 Main) — [ACL Anthology](https://aclanthology.org/2026.acl-long.1709/); 시간 기반 개인화 메모리 TSM(ACL 2026 Findings) — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1496/); TiMem(ACL 2026 Findings, arXiv Comments) — [arXiv abs](https://arxiv.org/abs/2601.02845)
- 2026 하반기 메모리-그래프 프리프린트(모두 arXiv abs 확인, Comments에 채택 정보 없음 → Preprint): Entity-Memory Graph Retrieval Improves Evidence Coverage in Long-Conversation QA(Shumao Sun, [2608.27925](https://arxiv.org/abs/2608.27925), 2026-08-28); RippleMem(Jingbo Ji et al., [2608.13334](https://arxiv.org/abs/2608.13334), 2026-08-13, 연상적 회상); Profile-Graph Memory for LLM Agents(Shengtong Zhu, [2607.19359](https://arxiv.org/abs/2607.19359), v1 2026-06-01, ID는 2607, MemHop 벤치마크 포함); Event-Centric Memory with Query-Aware Graph Augmentation(QGMem, Yichen Liu et al., [2610.11920](https://arxiv.org/abs/2610.11920), 2026-10-08, Comments "Submitted to IEEE TPAMI"); SGMem: Sentence Graph Memory(Yaxiong Wu et al., [2509.21212](https://arxiv.org/abs/2509.21212), 2025-09-25)
- 그래프 검색의 정책 학습: GraphRAG-R1(WWW 2026) — [arXiv abs](https://arxiv.org/abs/2507.23581); Graph-R1(ICML 2026) — [arXiv abs](https://arxiv.org/abs/2507.21892); Deep GraphRAG(DW-GRPO로 1.5B 통합 모델 학습, Preprint) — [arXiv abs](https://arxiv.org/abs/2601.11144); **HyperGraphPro: Progress-Aware Reinforcement Learning for Structure-Guided Hypergraph RAG**(Jinyoung Park et al., [2601.17755](https://arxiv.org/abs/2601.17755), v1 2026-01-25, v3 2026-11-05, Comments "In progress"). v1 제목은 "ProGraph-R1: Progress-aware Reinforcement Learning for Graph Retrieval Augmented Generation"으로, 이후 개명됨. 답 유도 목표 하이퍼엣지 커버리지에 따른 단계별 진척 보상을 부여 [✅ 초판의 fugumt 색인을 arXiv 1차로 대체]
- 연합/분산 GraphRAG: "Federated Graph Retrieval-Augmented Generation"(FedGRAG; Xiaobao Wang et al., [2610.30728](https://arxiv.org/abs/2610.30728), v1 2026-10-22, Preprint)은 경량 지식 인덱스로 관련 클라이언트를 찾는 Federated Knowledge Routing과, 검색된 증거 서브그래프를 증거 요약으로 바꿔 서버에서 재구성하는 통제된 공개(controlled-disclosure) 증거 구성을 제안 — [arXiv abs](https://arxiv.org/abs/2610.30728) [✅ 초판의 "재확인 실패" 해소]; SCOUT-RAG(Longkun Li et al., [2602.08400](https://arxiv.org/abs/2602.08400), 2026-02-09, 분산 도메인 에이전틱 Graph-RAG, 4개 협력 에이전트, Preprint) [✅]
- 경험 증강: LivingRAG — "Augmenting Graph RAG with Experience"(2026-08-26, Preprint) — [arXiv abs](https://arxiv.org/abs/2608.25960) [✅]
- (범위 외지만 신호) GraphRAG 보안: LogicPoison: Logical Attacks on Graph Retrieval-Augmented Generation(Yilin Xiao et al., [2604.02954](https://arxiv.org/abs/2604.02954), v1 2026-04-03) — ⚠️ 베뉴 "ACL 2026"(GraphRAG-Benchmark repo News "accepted by ACL'26"; arXiv Comments 없음, ACL 2026 accepted list 접속 불가로 Main/Findings 미확인. 초판의 "Main 목록 등재"는 재확인 불가). 멀티모달 GraphRAG: MegaRAG(Chi-Hsiang Hsiao et al., **ACL 2026 Main**, 2026.acl-long.2218) — [ACL Anthology](https://aclanthology.org/2026.acl-long.2218/) [✅]; HVM-GraphRAG(Xin He et al., [2607.24861](https://arxiv.org/abs/2607.24861), 2026-07-26, Preprint) [✅]

### Inferences
- 2026년의 무게중심은 "더 큰 KG를 더 정교하게 추출"이 아니라 "더 싸고, 더 세밀한 단위로, 더 잘 연결된 인덱스 + 학습된 순회 정책" 쪽으로 이동 중이다.
- 에이전트 메모리 연구가 GraphRAG 기법(PPR, 엔티티 그래프, 계층 통합)을 흡수하면서 "memory-augmented RAG"와 "GraphRAG"가 사실상 하나의 설계 공간으로 수렴하고 있다(MAGMA, HippoRAG 2, ComoRAG). ACL 2026에 메모리-그래프 논문이 Main(MAGMA, Does Memory Need Graphs?, APEX-MEM)과 Findings(TSM, TiMem)에 다수 채택된 것도 같은 흐름이다.
- 연합/분산 GraphRAG(FedGRAG, SCOUT-RAG)와 그래프 대상 공격(LogicPoison)의 등장은 GraphRAG가 엔터프라이즈 배포 단계로 들어섰음을 시사한다(추정).

### Gaps
- 2026 하반기(7–11월) 프리프린트들은 초록만 확인했고 본문 결과는 대조하지 않았다. NeurIPS 2026 채택 결과(그래프 RAG 관련)도 확인하지 못했다(neurips.cc·OpenReview 접속 불가).
- LogicPoison의 ACL 2026 Main/Findings 구분은 미확인.
- ICLR 2027 제출작은 심사 중이므로 베뉴 표기 대상이 아니다.

## Verification log (2026-11-19, 1차 출처 재검증)

**존재 확인**: 본문에 arXiv ID가 있는 논문 38편과 추가 5편(TiMem, LogicPoison, APEX-MEM, MemORAI, HVM-GraphRAG)의 `arxiv.org/abs` 페이지를 모두 내려받아 제목·저자·v1 날짜·Comments·초록을 대조했다. ACL Anthology 엔트리 9건(LightRAG, Does Memory Need Graphs?, MAGMA, STEM, ZoomRAG, TRA, TSM, MegaRAG, Atom-RAG)을 메타데이터로 확인했다. arXiv/OpenReview/ACL Anthology 어디에서도 찾지 못한 논문은 SeRAG 1편이며 목록에서 제거했다.

**정정 사항**
1. ComoRAG: 1저자 미확인 → **Juyuan Wang** 확인. v1 2025-08-14.
2. Does Memory Need Graphs?: 베뉴 Unverified → **ACL 2026 Main**(2026.acl-long.1232, pp. 26758–26782, Anthology 직접 확인). 설계 축 "추출·인덱싱·검색·QA" → 초록 기준 "표현·조직·유지·검색"으로 정정(Q1·Q2).
3. Graph-R1(2507.21892): Unverified → **ICML 2026**(arXiv Comments, Journal-ref, PDF PMLR 306 각주).
4. RAG vs. GraphRAG(2502.11371): Preprint → **KDD 2026 ⚠️**(v4 2026-11-07 PDF의 ACM 블록과 DOI 10.1145/3770855.3817575, ACM DL 미확인). Q3의 단일 홉/추론 구분 주장은 v4 초록에 없음을 명시.
5. AtomicRAG: 최초 공개 "2026-04" → arXiv v1 **2026-02-10**(ID 2604, PDF "Preprint. April 24, 2026"). 제목 하이픈을 "Atom-Entity"로 정정. 요약에서 초록에 없는 "포함·관련성·동의어 엣지, Atomic Sieve, 질의 분해"를 삭제하고 "관계 존재 여부만 표시하는 엣지, PPR+관련성 필터링, 이론 분석+5개 벤치마크"로 교체. 동명 유사 논문 Atom-RAG(EMNLP 2026 Findings)와 구분.
6. GraphRAG-Bench(2506.05690): 요약에서 초록에 없는 "정보 밀도가 다른 코퍼스"를 삭제하고 4개 과제 유형·전 파이프라인 평가로 정정. 구체적 결론은 본문/README 기준임을 명시. ICLR 2026 근거를 v3 PDF 머리말로 보강. v3 날짜 2026-02-22 확정. 공저자 6인 추가.
7. GraphRAG-R1: "rollout 검색 강화 GRPO" → "rollout-with-thinking 지원 수정 GRPO". PRA·CAF 약어를 풀어 쓰고 3단계 훈련으로 정정. 초록에 없는 "HotpotQA·2Wiki·MuSiQue 어댑터 공개" 삭제. ACM DOI는 미확인으로 표시.
8. LinearRAG: 요약에서 초록에 없는 "PPR"·데이터셋명을 삭제하고 "전역 중요도 집계", "추가 토큰 소비 없음", "4개 데이터셋"으로 정정. "77% 인덱싱 시간 절감"은 PDF 본문에서 확인([Q✓]). ICLR 2026은 1차 페이지 접속 불가로 ⚠️(간접 근거) 표시.
9. STEM: "schema-guided subgraph matching"(초록에 없는 표현) → 원문 "schema-guided graph search task". Semantic-to-Structural Projection 단계를 추가하고, Comments의 Oral 표기와 Anthology 쪽수(pp. 7252–7285)를 추가.
10. Ψ-RAG: 초록에 없는 "청크 유사도 순위화" 삭제. v1 2026-05-01 확정. ICML 2026 근거를 Comments + PDF 각주로 보강.
11. HippoRAG 2 / A-MEM / LeanRAG / ComoRAG / LightRAG / MAGMA: v1 날짜를 일(day) 단위로 확정. 요약은 대체로 정확했으나 초록 근거가 없는 평가성 문구("대표 오픈소스 구현", "대표 기준선")를 요약에서 빼고, 표준 RAG 대비 사실 기억 저하 문제(HippoRAG 2) 등 초록 핵심을 보강. A-MEM의 arXiv 표기 제목은 "A-MEM".
12. 보조 후보: AGRAG → **ICDE 2026**(Comments), "TF-IDF" → "통계 기반 방법". GraFine → **CIKM 2026**(Comments). EraRAG → Comments "Under review"(Preprint). Temporal Semantic Memory의 정식 제목에 "Beyond Dialogue Time:" 추가. ZoomRAG·TRA·MegaRAG의 저자·쪽수를 Anthology로 확정.
13. **SeRAG("Token-Free Hierarchical Indexing for RAG beyond LLM-based Summarization") 제거**: icml.cc 접속 불가, arXiv 제목 검색·웹 검색에서 미발견 → Unverified. Q1·Q2·Q3·Q4의 모든 언급을 삭제했다.
14. 2601.17755: "ProGraph-R1"(fugumt 색인) → 현재 arXiv 제목 **HyperGraphPro**(v1 제목이 ProGraph-R1), Jinyoung Park et al.
15. FedGRAG(2610.30728): "재확인 실패" → arXiv abs로 저자(Xiaobao Wang et al.)·v1 2026-10-22·초록(경량 지식 인덱스 라우팅, controlled-disclosure 증거 구성)을 확인.
16. TreeHop(2504.20114): "top-5 RAPTOR가 직접 검색보다 뒤처진다"는 주장을 v1·v3 PDF에서 찾지 못해 ❌ 삭제하고, v3 본문의 실제 비교 내용으로 교체.
17. TiMem·APEX-MEM·MemORAI("2차 인용만") → arXiv에서 존재와 Comments(ACL 2026 Findings / "Accepted to ACL 2026 Mains" / "ACL Findings")를 확인. LogicPoison은 arXiv 2604.02954로 존재를 확인했고, 베뉴는 ACL 2026(Main/Findings 미확인 ⚠️)으로 하향.
18. 2609.08599 서베이: 정식 제목·저자·**ICKG 2026**(Comments)을 추가. 초판에서 이 논문을 "Does Memory Need Graphs?"의 ACL 채택 근거로 쓰던 2차 인용은 Anthology 직접 확인으로 대체.
19. GraphRAG-Bench 서론의 "13.4% lower"(Han et al. 인용)는 v3 PDF에서 확인. 2차 출처 수치 중 53.4는 PDF에서 찾지 못해 ⚠️ 유지.

**여전히 Unverified/⚠️**: LinearRAG ICLR 2026(간접), RAG vs. GraphRAG KDD 2026(PDF ACM 블록만), LogicPoison의 Main/Findings 구분, GraphRAG-R1 ACM DOI, PMLR/AAAI OJS 쪽수, OpenReview decision 라벨(Poster/Oral 등) 전반. 원인은 모두 OpenReview·학회 사이트·doi.org가 이 세션 네트워크에서 차단되었기 때문이다.
