# Agentic RAG & 추론 통합 검색(Reasoning-Integrated Retrieval) — 2026-11-19 기준 연구 노트

> 범위: RL로 학습한 검색 에이전트(추론과 검색을 번갈아 수행), deep-research 및 장기(long-horizon) 웹/검색 에이전트, 반복·멀티홉 검색과 추론 모델 기반 검색 계획, 적응형 검색(검색 시점·여부·대상 결정), 멀티에이전트 RAG, RAG의 test-time scaling.
> 범위 밖(다른 조사자 담당): 그래프/KG RAG, 리트리버·리랭커·압축·효율, 멀티모달 RAG, 견고성·보안·평가.

## Q1. 2025–2026년 agentic/reasoning RAG에서 가장 영향력 있고 최신인 논문은 무엇이며, 각각 어느 학회에 채택되었나?

### Takeaway
2025년 초 3편의 기초 논문이 이 분야의 기본 틀을 세웠다. 프롬프트 기반으로 대형 추론 모델(LRM)에 검색을 결합한 Search-o1(EMNLP 2025 Main), RL로 추론과 검색을 교차 학습시킨 Search-R1(COLM 2025), 실제 웹에서 RL을 수행한 DeepResearcher(EMNLP 2025 Main)다. 2026년 채택 논문의 무게중심은 다음으로 옮겨 갔다.
- 보상 설계: 장문 rubric 보상, 과정 보상과 credit assignment
- 시뮬레이션 학습 환경
- 과잉 검색 억제와 적응형 검색
- 장기 컨텍스트·메모리 관리와 test-time scaling
- 멀티에이전트 공동 최적화
- 검색 인터페이스 재설계

아래 12편은 모두 학회 채택 여부를 1차 데이터 또는 OpenReview 미러로 확인했다.

### Cited Findings

#### 검증 방법과 한계 (먼저 읽을 것)
- 이 세션에서는 arxiv.org, export.arxiv.org, openreview.net, aclanthology.org에 직접 접속할 수 없었다(프록시 403 / DNS 차단). 그래서 다음 경로로 확인했다.
  - (a) ACL/EMNLP/EACL의 Main·Findings 여부: ACL Anthology 공식 소스 데이터(`data/xml/*.xml`)로 확인했다. 이는 Anthology 사이트를 생성하는 1차 데이터다. — [acl-org/acl-anthology](https://github.com/acl-org/acl-anthology)
  - (b) NeurIPS 2025/2026, ICML 2026, COLM 2025 결정: OpenReview 결정을 미러링한 Paper Copilot JSON의 `status` 필드(Poster/Spotlight/Oral/Reject)로 확인했다. OpenReview 원문 페이지는 직접 열람하지 못했으므로, 미러 기반 확인이라는 점을 명시한다. — [papercopilot/paperlists](https://github.com/papercopilot/paperlists)
  - (c) ICLR 2026: Paper Copilot 파일이 Git LFS라 받을 수 없었다. iclr.cc 공식 가상 포스터 페이지(웹 검색 결과)로 개별 확인했다. — [ICLR 2026 poster page 예시](https://iclr.cc/virtual/2026/poster/10008772)
- 요약과 인용구는 학회판 초록(ACL Anthology XML의 `<abstract>`, OpenReview 초록 미러)을 기준으로 작성했다. arXiv v1 초록과 수치가 다를 수 있다. 예를 들어 Search-R1의 "41%/20%"는 COLM 판 초록의 수치다. — [papercopilot/paperlists](https://github.com/papercopilot/paperlists)
- 최초 공개일은 arXiv ID의 YYMM(최초 제출 월)로 표기했다. arXiv abs 페이지에 접속할 수 없어 정확한 일자(day)는 확인하지 못했다(Gaps 참조).
- ACL Anthology URL은 Anthology ID 규칙 `{연도}.{학회}-{볼륨}.{번호}`에 따라 XML의 collection/volume/paper ID에서 조립했다.

#### 큐레이션 목록 (12편; [F] = 2025년 초 기초(foundational) 논문)

**[1][F] Search-o1: Agentic Search-Enhanced Large Reasoning Models** — Xiaoxi Li et al.
- arXiv: [2501.05366](https://arxiv.org/abs/2501.05366) (ID 출처: [Awesome-RAG-Reasoning](https://github.com/DavidZWZ/Awesome-RAG-Reasoning)) | 최초 공개: 2025-01 (ID 기준)
- 학회: **EMNLP 2025 Main** — [2025.emnlp-main.276](https://aclanthology.org/2025.emnlp-main.276/) (Anthology XML로 확인)
- 요약 1: o1류 대형 추론 모델(LRM)은 긴 추론 도중 지식이 부족해 불확실성과 오류를 자주 겪는다. 이를 해결하려고, 불확실한 지식 지점에서 동적으로 외부 지식을 검색하는 agentic RAG 워크플로를 추론 과정에 통합하고, 장황한 검색 문서를 분석·정제한 뒤 추론 체인에 주입하는 Reason-in-Documents 모듈을 제안한다.
- 요약 2: 과학·수학·코딩 복잡 추론 과제와 6개 오픈도메인 QA 벤치마크에서 강한 성능을 보였다. LRM의 신뢰성을 높이고 고급 deep research 시스템으로 가는 길을 제시한 것이 기여다.
- 인용: "integrates an agentic search workflow into the reasoning process"
- 하위분류: 프롬프트/워크플로 기반 agentic RAG. Awesome-RAG-Reasoning도 "Single-Agent › Prompting"으로 분류한다. — [Awesome-RAG-Reasoning](https://github.com/DavidZWZ/Awesome-RAG-Reasoning)

**[2][F] Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning** — Bowen Jin et al.
- arXiv: [2503.09516](https://arxiv.org/abs/2503.09516) (ID 출처: [Awesome-RAG-Reasoning](https://github.com/DavidZWZ/Awesome-RAG-Reasoning)) | 최초 공개: 2025-03 (ID 기준)
- 학회: **COLM 2025** (Paper Copilot status = Poster; COLM 2025 채택 논문은 모두 Poster로 표기됨) — [OpenReview Rwhi91ideu](https://openreview.net/forum?id=Rwhi91ideu) (via [Paper Copilot colm2025.json](https://github.com/papercopilot/paperlists))
- 요약 1: 추론 능력이 있는 LLM에게 프롬프트만으로 검색엔진을 쓰게 하면 상호작용 방법을 몰라 최적에 못 미친다. 이 문제를 풀기 위해 LLM이 단계별 추론 중에 실시간 검색과 함께 (여러 개의) 검색 쿼리를 스스로 생성하도록 RL로 학습시킨다. 안정적인 학습을 위해 검색된 토큰을 마스킹하고, 단순한 결과(outcome) 기반 보상을 쓴다.
- 요약 2: 7개 QA 데이터셋에서 같은 설정의 RAG 베이스라인보다 Qwen2.5-7B는 41%, Qwen2.5-3B는 20% 향상했다(COLM 판 초록 기준). RL 최적화 방법, LLM 선택, 응답 길이 동역학에 대한 경험적 통찰도 함께 제시한다.
- 인용: "leveraging retrieved token masking for stable RL training and a simple outcome-based reward function"
- 하위분류: RL 학습 검색 에이전트. 이후 대부분의 2025–2026 연구가 비교 기준으로 사용한다.

**[3][F] DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments** — Yuxiang Zheng et al.
- arXiv: [2504.03160](https://arxiv.org/abs/2504.03160) (ID 출처: [Awesome-RAG-Reasoning](https://github.com/DavidZWZ/Awesome-RAG-Reasoning)) | 최초 공개: 2025-04 (ID 기준)
- 학회: **EMNLP 2025 Main** — [2025.emnlp-main.22](https://aclanthology.org/2025.emnlp-main.22/)
- 요약 1: 기존 deep research 방법(취약한 프롬프트 엔지니어링, 통제된 환경에서의 RAG 기반 RL)은 현실 세계의 복잡성을 담지 못한다. 이를 해결하려고, 실제 웹 검색과 상호작용하는 환경에서 RL을 스케일링해 deep research 에이전트를 end-to-end로 학습하는 첫 종합 프레임워크를 제안한다. 다양한 웹페이지 구조에서 정보를 추출하는 브라우징 에이전트를 둔 멀티에이전트 구조다.
- 요약 2: 오픈도메인 리서치 과제에서 프롬프트 엔지니어링 기반 대비 최대 28.9점, RAG 기반 RL 에이전트 대비 최대 7.2점 향상했다. 계획, 교차검증, 자기성찰을 통한 방향 전환, 확답할 수 없을 때 정직하게 답하는 행동이 학습 과정에서 창발했다.
- 인용: "end-to-end training in real-world web environments is fundamental for developing robust research capabilities"
- 하위분류: Deep-research / 장기 웹 에이전트 (실제 웹 환경 RL)

**[4] DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research** — Rulin Shao et al.
- arXiv: [2511.19399](https://arxiv.org/abs/2511.19399) (웹 검색 결과로 arXiv URL·제목 일치 확인, v1–v3 존재) | 최초 공개: 2025-11 (ID 기준)
- 학회: **ICML 2026 (Spotlight)** — [OpenReview 97NEP1pyS3](https://openreview.net/forum?id=97NEP1pyS3) (via [Paper Copilot icml2026.json](https://github.com/papercopilot/paperlists))
- 요약 1: 대부분의 오픈 deep research 에이전트는 검증 가능한 단답형 QA로 RLVR 학습을 받기 때문에, 현실적인 장문 과제로 확장되지 않는다. 이를 해결하려고, 학습 중에 루브릭을 구성·유지하며 정책 모델과 공진화(co-evolve)시키는 RLER(Reinforcement Learning with Evolving Rubrics)를 제안하고 DR Tulu-8B를 개발했다.
- 요약 2: 과학·헬스케어·일반 도메인의 장문 deep research 벤치마크 4개에서 기존 오픈 에이전트보다 평균 15.6% 높고(Tongyi DR 대비), 독점 에이전트와 같거나 더 높은(OpenAI DR 대비 평균 +0.7%) 성능을 냈다. 쿼리당 비용은 OpenAI DR보다 1000배 저렴하다. 장문 deep research용으로 직접 학습한 첫 완전 오픈 모델이다.
- 인용: "rubrics are constructed and maintained to co-evolve with the policy model during training"
- 하위분류: Deep-research 에이전트 / 보상 설계(rubric reward)

**[5] TreePS-RAG: Tree-based Process Supervision for Reinforcement Learning in Agentic RAG** — Tianhua Zhang et al.
- arXiv: [2601.06922](https://arxiv.org/abs/2601.06922) (웹 검색 결과로 확인) | 최초 공개: 2026-01 (ID 기준)
- 학회: **EMNLP 2026 Findings** — [2026.findings-emnlp.1007](https://aclanthology.org/2026.findings-emnlp.1007/)
- 요약 1: 결과 기반의 희소한 최종 보상만으로는 agentic RAG의 단계별 credit assignment가 약하다. 기존 과정 감독은 오프라인으로 만든 데이터(분포 이동 위험)나 비싼 중간 주석에 의존한다. 이를 해결하려고, agentic RAG의 추론을 각 단계가 노드인 롤아웃 트리로 모델링하고, 후손 노드 결과에 대한 몬테카를로 추정으로 단계 효용을 구하는 온라인 트리 기반 RL을 제안한다. 보상은 결과 보상만 유지한다.
- 요약 2: Search-R1과 비슷한 롤아웃 비용으로, 7개 멀티홉·일반 QA 벤치마크와 여러 모델 규모에서 결과 감독 RL과 기존 과정 감독 RL 방법을 일관되게 앞선다.
- 인용: "model agentic RAG reasoning as a rollout tree, where each reasoning step naturally maps to a node"
- 하위분류: RL 검색 에이전트 › 과정 보상 / credit assignment

**[6] SearchGym: Bootstrapping Real-World Search Agents via Cost-Effective and High-Fidelity Environment Simulation** — Xichen Zhang et al.
- arXiv: [2601.14615](https://arxiv.org/abs/2601.14615) (웹 검색 결과로 확인; 2026년 1월 arXiv 목록에 게재) | 최초 공개: 2026-01 (ID 기준)
- 학회: **ACL 2026 Main** (Long) — [2026.acl-long.848](https://aclanthology.org/2026.acl-long.848/)
- 요약 1: 검색 에이전트 RL 학습에는 딜레마가 있다. 상용 웹 API와 상호작용하면 비용이 지나치게 크고, 정적 스냅샷을 쓰면 데이터 불일치로 보상 신호가 오염된다. 이를 해결하려고, 검증 가능한 지식그래프와 그에 정렬된 문서 코퍼스를 생성해 모든 과제가 사실에 근거하고 반드시 풀 수 있도록 만든 시뮬레이션 환경 SearchGym과, 커리큘럼 학습법 SearchGym-RL을 제안한다.
- 요약 2: Llama·Qwen 계열에서 강한 Sim-to-Real 일반화를 보였다. SearchGym에서 학습한 Qwen2.5-7B-Base는 9개 벤치마크에서 웹을 쓰는 ASearcher 베이스라인보다 평균 상대 10.6% 높다.
- 인용: "high-fidelity simulation serves as a scalable and highly cost-effective methodology"
- 하위분류: RL 검색 에이전트 › 시뮬레이션 학습 환경 (ZeroSearch 계열의 후속 방향)

**[7] Over-Searching in Search-Augmented Large Language Models** — Roy Xie et al.
- arXiv: [2601.05503](https://arxiv.org/abs/2601.05503) (ID 출처: [Awesome-RAG-Reasoning](https://github.com/DavidZWZ/Awesome-RAG-Reasoning)) | 최초 공개: 2026-01 (ID 기준)
- 학회: **EACL 2026 Main** (Long) — [2026.eacl-long.361](https://aclanthology.org/2026.eacl-long.361/)
- 요약 1: 검색 증강 LLM은 응답 품질이 나아지지 않는데도 검색 도구를 불필요하게 호출하는 과잉 검색(over-searching)을 보인다. 이 문제를 질의 유형, 모델 범주, 검색 조건, 다중 턴 대화의 여러 차원에서 체계적으로 평가했다.
- 요약 2: 검색은 답할 수 있는 질의의 정확도는 높이지만 답할 수 없는 질의에서는 기권(abstention)을 해친다. 과잉 검색은 복잡한 추론 모델과 deep research 시스템에서 더 심하고, 잡음 섞인 검색에서 악화되며, 다중 턴에서 누적된다. 비용과 성능의 트레이드오프를 재는 지표 TPC(Tokens Per Correctness)와 OverSearchQA 벤치마크를 공개했다.
- 인용: "unnecessarily invoking search tool even when it does not improve response quality"
- 하위분류: 적응형 검색 / 검색 호출 효율

**[8] AdaSearch: Balancing Parametric Knowledge and Search in Large Language Models via Reinforcement Learning** — Tzu-Han Lin et al.
- arXiv: [2512.16883](https://arxiv.org/abs/2512.16883) (웹 검색 결과에서 얻은 ID로, 저자 목록은 Anthology 레코드와 일치한다. arXiv 원문 페이지는 직접 확인하지 못함) | 최초 공개: 2025-12 (ID 기준)
- 학회: **EMNLP 2026 Main** — [2026.emnlp-main.2696](https://aclanthology.org/2026.emnlp-main.2696/)
- 요약 1: RL 검색 에이전트가 내부 파라메트릭 지식과 외부 검색 사이의 균형을 적응적으로 잡지 못하는 문제를 다룬다. 검색에 지나치게 의존하면 비용이 늘고 잡음·악성 콘텐츠에 노출되며, 파라메트릭 지식에만 의존하면 환각이 생긴다. 저자들은 F1 기반 결정 지표로 평가를 다시 검토한 뒤, 문제 해결과 검색 여부 결정을 분리한 2단계 결과 기반 RL 프레임워크 AdaSearch를 제안한다.
- 요약 2: 검색 결정 품질을 크게 높이고 불필요한 검색 호출을 줄였다. QA 정확도는 항상 검색하는 방식보다 약간 낮아지는 정도에 그쳤고, 결정 과정을 명시적이고 해석 가능하게 만들었다.
- 인용: "disentangles problem-solving from the decision to search"
- 하위분류: 적응형 검색(검색 여부 결정) × RL

**[9] FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents** — Chiwei Zhu et al.
- arXiv: [2602.01566](https://arxiv.org/abs/2602.01566) (ID 출처: [Awesome-RAG-Reasoning](https://github.com/DavidZWZ/Awesome-RAG-Reasoning)) | 최초 공개: 2026-02 (ID 기준)
- 학회: **ACL 2026 Main** (Long) — [2026.acl-long.288](https://aclanthology.org/2026.acl-long.288/)
- 요약 1: deep research의 긴 궤적은 컨텍스트 한계를 넘기 쉬워 증거 수집과 보고서 작성 예산을 압박하고 test-time scaling을 가로막는다. 이를 해결하려고, 파일 시스템을 지속적 작업공간으로 쓰는 이중 에이전트 프레임워크를 제안한다. Context Builder가 웹을 탐색하며 노트를 쓰고 원문을 계층적 지식베이스에 보관하면, Report Writer가 이를 사실 출처로 삼아 섹션별로 보고서를 작성한다.
- 요약 2: DeepResearch Bench와 DeepConsult에서 여러 백본 모델로 SOTA 보고서 품질을 달성했다. Context Builder에 배정한 계산량과 최종 보고서 품질이 양의 상관을 보여, 파일 시스템 패러다임에서도 test-time scaling이 작동함을 입증했다.
- 인용: "the file system serves as a durable external memory and a shared coordination medium across agents and sessions"
- 하위분류: Deep-research 에이전트 × test-time scaling × 멀티에이전트

**[10] FineVerify: Scaling Test-Time Compute with Fine-Grained Self-Verification for Agentic Search** — James Xu Zhao et al.
- arXiv: [2606.00660](https://arxiv.org/abs/2606.00660) (ID 출처: [Awesome-RAG-Reasoning](https://github.com/DavidZWZ/Awesome-RAG-Reasoning)) | 최초 공개: 2026-06 (ID 기준)
- 학회: **EMNLP 2026 Main** — [2026.emnlp-main.2599](https://aclanthology.org/2026.emnlp-main.2599/)
- 요약 1: agentic search에서 test-time compute를 늘려도 실패할 수 있다. 정답이 드물고, 점수 기반 선택이 모델의 보정(calibration)에 의존하기 때문이다. 이를 해결하려고, 질문을 검증 가능한 하위 질문으로 분해하고 샘플링한 후보 답을 하위 질문별로 검증한 뒤 점수를 집계해 고르는 세밀한 자기검증 프레임워크를 제안한다.
- 요약 2: 4개 agentic search 벤치마크와 2개 모델에서 표준 스케일링 기법을 일관되게 앞섰다. 궤적 4개만으로 GPT-5-mini는 8.2점, Gemini-3-flash는 평균 5.6% 향상했고, 샘플 12개로 GPT-5-mini가 BrowseComp-Plus에서 GPT-5를 넘어섰다.
- 인용: "decomposes each question into checkable sub-questions"
- 하위분류: 프롬프트/추론 시점 기법 × test-time scaling (학습 불필요)

**[11] JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG** — Yiqun Chen et al.
- arXiv: [2601.21916](https://arxiv.org/abs/2601.21916) (웹 검색 결과로 확인) | 최초 공개: 2026-01 (ID 기준)
- 학회: **ICML 2026 (Poster)** — [OpenReview KQT6pf4Mva](https://openreview.net/forum?id=KQT6pf4Mva) (via [Paper Copilot icml2026.json](https://github.com/papercopilot/paperlists))
- 요약 1: 동적 agentic RAG의 기존 방식은 둘 중 하나다. 고정 그래프 구조 안에서 모듈을 함께 최적화하거나, 동적으로 계획하되 실행기를 동결된 블랙박스 도구로 둔다. 그 결과 정교한 계획이 실행되지 못하는 "전략-운영 불일치"가 생긴다. 이를 해결하려고, 백본을 공유하는 협력 멀티에이전트 팀으로 계획(planner)과 실행(executor)을 결과 보상 기반으로 end-to-end 공동 최적화하는 JADE를 제안한다.
- 요약 2: planner는 executor의 능력 범위 안에서 계획하고, executor는 전략 의도에 맞게 진화하는 co-adaptation이 일어난다. 이를 통해 분리되어 있던 모듈을 시너지를 내는 시스템으로 바꿔 성능을 높이고, 동적 워크플로 오케스트레이션으로 효율과 효과의 균형을 유연하게 조절한다. 초록에는 구체적 수치가 없다.
- 인용: "the planner learns to operate within executor capability boundaries"
- 하위분류: 멀티에이전트 RAG × RL 공동 최적화. 2026년 SoK 서베이도 이 불일치를 다루는 사례로 인용한다. — [SoK: Agentic RAG (2603.07379)](https://arxiv.org/abs/2603.07379)

**[12] Beyond Semantic Similarity: Rethinking Retrieval for Agentic Search via Direct Corpus Interaction** — Zhuofeng Li et al.
- arXiv: [2605.05242](https://arxiv.org/abs/2605.05242) (웹 검색 결과와 Awesome-RL-based-Agentic-Search 목록으로 확인) | 최초 공개: 2026-05 (ID 기준)
- 학회: **NeurIPS 2026 (Spotlight)** — [OpenReview 42dIxEhArH](https://openreview.net/forum?id=42dIxEhArH), [NeurIPS virtual poster 155649](https://neurips.cc/virtual/2026/poster/155649) (via [Paper Copilot nips2026.json](https://github.com/papercopilot/paperlists))
- 요약 1: 고정된 유사도 기반 top-k 검색 인터페이스는 정확한 어휘 매칭, 약한 단서들의 결합, 국소 맥락 검증, 다단계 가설 수정이 필요한 agentic search에서 병목이 된다. 이를 해결하려고, 에이전트가 임베딩·벡터 인덱스·검색 API 없이 grep, 파일 읽기, 셸 명령 같은 범용 터미널 도구로 원시 코퍼스를 직접 탐색하는 Direct Corpus Interaction(DCI)을 연구한다.
- 요약 2: BRIGHT·BEIR 검색 벤치마크와 BrowseComp-Plus의 end-to-end agentic search에서 강력한 sparse·dense·reranking 베이스라인을 크게 앞섰고, 기존 시맨틱 리트리버 없이도 멀티홉 QA에서 높은 정확도를 냈다. 검색 품질이 추론 능력뿐 아니라 코퍼스 인터페이스가 얼마나 풍부한 상호작용을 지원하는지에도 달려 있음을 보였다.
- 인용: "retrieval quality depends not only on reasoning ability, but also on how richly the corpus interface supports interaction"
- 하위분류: 신흥 방향 — 에이전트용 검색 인터페이스 재설계 (리트리버 연구와 경계에 있음)

#### 같은 방식으로 학회 채택을 확인한 추가 대표 논문 (제목·라벨만; 분류는 Q2에서 사용)
- NeurIPS 2025 (Paper Copilot 미러):
  - ReSearch (Mingyang Chen et al.) — Poster
  - WebThinker (Xiaoxi Li et al.) — Poster
  - WebDancer (Jialong Wu et al.) — Poster
  - Chain-of-Retrieval Augmented Generation / CoRAG (Liang Wang et al.) — Poster
  - Improving RAG through Multi-Agent RL / MMOA-RAG (Yiqun Chen et al.) — Poster
  - Search and Refine During Think / AutoRefine (Yaorui Shi et al.) — Poster
  - Process vs. Outcome Reward: Which is Better for Agentic RAG RL (Wenlin Zhang et al.) — Poster
  - Iterative Self-Incentivization Empowers LLMs as Agentic Searchers (Zhengliang Shi et al.) — Poster
  - AceSearcher (Ran Xu et al.) — Spotlight
  - 출처: [papercopilot/paperlists nips2025.json](https://github.com/papercopilot/paperlists)
- EMNLP 2025 Main:
  - StepSearch (Xuhui Zheng et al.) — [2025.emnlp-main.1106](https://aclanthology.org/2025.emnlp-main.1106/)
  - s3: You Don't Need That Much Data to Train a Search Agent via RL (Pengcheng Jiang et al.) — [2025.emnlp-main.1095](https://aclanthology.org/2025.emnlp-main.1095/)
  - EvolveSearch — [2025.emnlp-main.663](https://aclanthology.org/2025.emnlp-main.663/)
  - Search Wisely: Mitigating Sub-optimal Agentic Searches By Reducing Uncertainty — [2025.emnlp-main.998](https://aclanthology.org/2025.emnlp-main.998/)
- ICLR 2026 (Poster): Tree Search for LLM Agent Reinforcement Learning / Tree-GRPO (Yuxiang Ji et al., arXiv [2509.21240](https://arxiv.org/abs/2509.21240)). Poster Session 6, 2026-04-25. — [iclr.cc poster](https://iclr.cc/virtual/2026/poster/10008772)
- ICML 2026 (Poster) — 출처: [papercopilot/paperlists icml2026.json](https://github.com/papercopilot/paperlists)
  - Stratified GRPO (Mingkang Zhu et al., arXiv [2510.06214](https://arxiv.org/abs/2510.06214), 2025년 10월 제출로 웹 검색 결과에서 확인) — [OpenReview ZX2lOgKOTm](https://openreview.net/forum?id=ZX2lOgKOTm)
  - DECOR (Ruiqing Chen et al.) — [OpenReview dkRAVqr347](https://openreview.net/forum?id=dkRAVqr347)
  - ReSeek — [OpenReview SeO8pLS44H](https://openreview.net/forum?id=SeO8pLS44H)
  - RE-TRAC — [OpenReview wjNTbZKIvC](https://openreview.net/forum?id=wjNTbZKIvC)
  - IntentRL — [OpenReview rWSwu2Req3](https://openreview.net/forum?id=rWSwu2Req3)
  - Optimizing Agentic Reasoning with Retrieval via Synthetic Semantic Information Gain Reward — [OpenReview mUU7k4bN97](https://openreview.net/forum?id=mUU7k4bN97)
- NeurIPS 2026 (Poster) — 출처: [papercopilot/paperlists nips2026.json](https://github.com/papercopilot/paperlists)
  - PiCA: Pivot-Based Credit Assignment (Dongyi Liu et al.) — [OpenReview 4CNCefRkpK](https://openreview.net/forum?id=4CNCefRkpK)
  - Beyond Outcome Rewards: Step-Level Self-Distilled PO for Deep Search Agents — [B1ywhnzK7h](https://openreview.net/forum?id=B1ywhnzK7h)
  - Quest: Training Frontier Deep Research Agents with Fully Synthetic Tasks — [eKFW4EIOJx](https://openreview.net/forum?id=eKFW4EIOJx)
  - Deep Research as Rubric — [JmP1iLzzlJ](https://openreview.net/forum?id=JmP1iLzzlJ)
  - RICE-PO — [2fIYREq48r](https://openreview.net/forum?id=2fIYREq48r)
  - SD-Search — [50CwcISp9W](https://openreview.net/forum?id=50CwcISp9W)
  - SpecHop — [iS6Bu6eqWc](https://openreview.net/forum?id=iS6Bu6eqWc)
  - CARD — [Q15eHqL9AR](https://openreview.net/forum?id=Q15eHqL9AR)
  - Retriever-Free Retrieval-Augmented Reasoning via Corpus-Traversing MCTS — [vi37A4okc6](https://openreview.net/forum?id=vi37A4okc6)
- ACL 2026 Main (Long):
  - End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based RL / MHGPO (Guanzhong Chen et al.; arXiv [2506.02718](https://arxiv.org/abs/2506.02718), v1 제목은 "Heterogeneous Group-Based Reinforcement Learning for LLM-based Multi-Agent Systems") — [2026.acl-long.1399](https://aclanthology.org/2026.acl-long.1399/)
  - MemSearch-o1 — [2026.acl-long.41](https://aclanthology.org/2026.acl-long.41/)
  - S2G-RAG — [2026.acl-long.1185](https://aclanthology.org/2026.acl-long.1185/)
  - Adaptive Retrieval for Reasoning (REPAIR) — [2026.acl-long.1734](https://aclanthology.org/2026.acl-long.1734/)
  - Trust Within? Seek Beyond? Knowledge Boundary Aware PO for Agentic Search — [2026.acl-long.1276](https://aclanthology.org/2026.acl-long.1276/)
  - Reinforcing Agentic Search Via Reward Density Optimization — [2026.acl-long.467](https://aclanthology.org/2026.acl-long.467/)
  - Chaining the Evidence: … Citation-Aware Rubric Rewards — [2026.acl-long.950](https://aclanthology.org/2026.acl-long.950/)
  - PRInTS: Reward Modeling for Long-Horizon Information Seeking — [2026.acl-long.1574](https://aclanthology.org/2026.acl-long.1574/)
  - D^2Plan: Dual-Agent Dynamic Global Planning — [2026.acl-long.216](https://aclanthology.org/2026.acl-long.216/)
- ACL 2026 Findings:
  - BrowseConf: Confidence-Guided Test-Time Scaling for Web Agents — [2026.findings-acl.21](https://aclanthology.org/2026.findings-acl.21/)
  - CriticSearch — [2026.findings-acl.596](https://aclanthology.org/2026.findings-acl.596/)
  - MemSearcher — [2026.findings-acl.736](https://aclanthology.org/2026.findings-acl.736/)
  - AutoSearch: Adaptive Search Depth for Efficient Agentic RAG via RL — [2026.findings-acl.1399](https://aclanthology.org/2026.findings-acl.1399/)
  - BAPO: Boundary-Aware Policy Optimization — [2026.findings-acl.1814](https://aclanthology.org/2026.findings-acl.1814/)
  - Agentic-R: Learning to Retrieve for Agentic Search — [2026.findings-acl.785](https://aclanthology.org/2026.findings-acl.785/)
- EMNLP 2026 Main:
  - Fake It Till You Make It: Training Deep-Search Agents in Synthetic Search Environments (SynSearch) — [2026.emnlp-main.928](https://aclanthology.org/2026.emnlp-main.928/)
  - Dense Process Supervision for Search Agents via Fact Utility Estimation — [2026.emnlp-main.1050](https://aclanthology.org/2026.emnlp-main.1050/)
  - OpenResearcher — [2026.emnlp-main.42](https://aclanthology.org/2026.emnlp-main.42/)
  - DeepSearch-World — [2026.emnlp-main.1117](https://aclanthology.org/2026.emnlp-main.1117/)
  - CoPRM — [2026.emnlp-main.65](https://aclanthology.org/2026.emnlp-main.65/)
  - To Answer or to Abstain — [2026.emnlp-main.161](https://aclanthology.org/2026.emnlp-main.161/)
  - P-Search — [2026.emnlp-main.570](https://aclanthology.org/2026.emnlp-main.570/)
  - KnowGPO — [2026.emnlp-main.2357](https://aclanthology.org/2026.emnlp-main.2357/)
  - DEEPRUBRIC — [2026.emnlp-main.2267](https://aclanthology.org/2026.emnlp-main.2267/)
  - Aggregate While Searching — [2026.emnlp-main.503](https://aclanthology.org/2026.emnlp-main.503/)
- EMNLP 2026 Findings:
  - HALT: Verification-Aware Stopping — [2026.findings-emnlp.343](https://aclanthology.org/2026.findings-emnlp.343/)
  - One Policy, Any Budget — [2026.findings-emnlp.1857](https://aclanthology.org/2026.findings-emnlp.1857/)
  - Reinforced Internal-External Knowledge Synergistic Reasoning — [2026.findings-emnlp.2184](https://aclanthology.org/2026.findings-emnlp.2184/)
  - Beyond Confidence: Test-Time Scaling for Multi-Turn Search Agents — [2026.findings-emnlp.870](https://aclanthology.org/2026.findings-emnlp.870/)
  - STAMP — [2026.findings-emnlp.637](https://aclanthology.org/2026.findings-emnlp.637/)
  - APEX-Searcher — [2026.findings-emnlp.1538](https://aclanthology.org/2026.findings-emnlp.1538/)
  - Negative Advantage Is a Double-Edged Sword — [2026.findings-emnlp.1874](https://aclanthology.org/2026.findings-emnlp.1874/)
  - DeepDiver-DR — [2026.findings-emnlp.1154](https://aclanthology.org/2026.findings-emnlp.1154/)
  - Beyond Turn Limits — [2026.findings-emnlp.2461](https://aclanthology.org/2026.findings-emnlp.2461/)

#### 학회 채택 여부를 확인하지 못한 영향력 있는 논문 → "Unverified"
- ZeroSearch(2505.04588), R1-Searcher(2503.05592), ASearcher "Beyond Ten Turns"(2508.07976), Search Self-Play(2510.18821), Tongyi DeepResearch(2510.24701), CoSearch(2604.17555), "To Search or Not to Search"(2602.03304)는 다음 데이터의 제목 검색에서 찾지 못했다: NeurIPS 2025/2026, ICML 2026, COLM 2025, ACL Anthology 2025(acl/emnlp/findings)·2026 전체. ICLR 2026, AAAI 2026, WWW 2026, SIGIR 2026, KDD 2026은 확인하지 못했다. 따라서 이 논문들의 학회 표기는 **Unverified**다. — [papercopilot/paperlists](https://github.com/papercopilot/paperlists), [acl-org/acl-anthology](https://github.com/acl-org/acl-anthology)
- Awesome-RAG-Reasoning(2차 출처)은 다음 SIGIR 2026 논문을 표기한다: "When to Retrieve During Reasoning: Adaptive Retrieval for Large Reasoning Models"(arXiv 2604.26649), CroSearch-R1(arXiv 2604.25182). SIGIR 공식 목록으로는 확인하지 못했으므로 **Unverified**로 둔다. — [Awesome-RAG-Reasoning](https://github.com/DavidZWZ/Awesome-RAG-Reasoning)

### Inferences
- 2025년 초 기초 논문 3편(Search-o1, Search-R1, DeepResearcher)은 이후 거의 모든 2026 논문이 비교 기준이나 출발점으로 삼는다(예: TreePS-RAG는 "Search-R1과 비슷한 롤아웃 비용", SearchGym은 ASearcher를 비교 기준으로 사용). 따라서 이 셋을 기초로 표시해 포함하는 것이 타당하다.
- 학회 채택 기준으로 보면, 2026년 이 하위 분야의 주 무대는 ACL 계열(ACL 2026, EMNLP 2026)과 ICML/NeurIPS 2026이다. Spotlight급은 deep research의 보상 설계(DR Tulu)와 검색 인터페이스 재설계(DCI)처럼 "패러다임 전환"형 논문이다.
- 많이 인용되는 산업계 시스템 보고서(Tongyi DeepResearch, ASearcher, ZeroSearch 등)는 확인 가능한 범위에서 학회 채택 기록이 없다. 영향력과 학회 채택이 꼭 일치하지는 않는다.

### Gaps
- 정확한 최초 공개 일자(day 단위): arxiv.org에 접속할 수 없어 arXiv ID의 YYMM(월)까지만 확인했다. 일자가 필요하면 arXiv abs 페이지의 "Submission history"로 재확인해야 한다.
- arXiv ID 일부(Search-o1, Search-R1, DeepResearcher, Over-Searching, FS-Researcher, FineVerify)는 awesome 리스트 링크에서 얻었다. AdaSearch(2512.16883)는 웹 검색 요약에서 얻었다. 모두 arXiv 원문 페이지로는 직접 대조하지 못했다(제목·저자는 학회 레코드와 일치).
- OpenReview 결정(COLM/ICML/NeurIPS)은 Paper Copilot 미러로 확인했고 OpenReview 원문은 열지 못했다. ICLR 2026은 Tree-GRPO 외에는 확인하지 못했다(LFS 파일 접근 불가). 그래서 ZeroSearch·ASearcher·Search Self-Play 등의 ICLR 2026 채택 여부는 미확인이다.
- AAAI 2026, WWW 2026, SIGIR 2026, KDD 2026 공식 채택 목록은 직접 확인하지 못했다.
- DECOR(ICML 2026)는 arXiv에서 찾지 못했다(웹 검색 결과 없음). 그래서 본 목록에서는 JADE로 대체했다.

## Q2. 이 하위 분야는 일반적으로 어떻게 세분되는가? (프롬프트 기반 워크플로 vs RL 학습 검색 에이전트 vs deep research vs 적응형 검색 등)

### Takeaway
기존 서베이는 두 축으로 나눈다. (1) RAG와 추론이 어느 방향으로 결합되는가(Reasoning-Enhanced RAG / RAG-Enhanced Reasoning / Synergized RAG-Reasoning), (2) 오케스트레이션을 어떻게 학습하는가(Prompting / SFT / RL, 단일 vs 멀티에이전트). 2026년 문헌을 반영하면 다음 7개 하위 범주가 실용적이다.
- A. 프롬프트/워크플로 기반 agentic RAG
- B. RL 학습 검색 에이전트. 다시 B1 기본 프레임워크, B2 RL 알고리즘·보상 설계, B3 학습 환경·데이터 합성으로 나뉜다.
- C. Deep-research / 장기 에이전트
- D. 적응형 검색과 검색 효율
- E. 멀티에이전트 RAG
- F. Test-time scaling
- G. 에이전트용 검색 인터페이스 재설계

### Cited Findings
- "Towards Agentic RAG with Deep Reasoning: A Survey of RAG-Reasoning Systems in LLMs"(arXiv 2507.09477)의 분류 체계는 공식 저장소에 반영되어 있다. 구성은 다음과 같다.
  - Reasoning-Enhanced RAG(Retrieval Optimization / Integration Enhancement / Generation Enhancement)
  - RAG-Enhanced Reasoning(External Knowledge Retrieval / In-context Retrieval)
  - Synergized RAG and Reasoning: Reasoning Workflow(Chain/Tree/Graph-based)와 Agentic Orchestration(Single-Agent: Prompting·SFT·RL / Multi-Agent)
  - 앞의 두 범주는 "one-way" 강화이고, Synergized 범주는 검색과 추론을 "iteratively" 수행해 서로를 강화한다.
  - 출처: [Awesome-RAG-Reasoning](https://github.com/DavidZWZ/Awesome-RAG-Reasoning)
- "A Comprehensive Survey on RL-based Agentic Search"(arXiv 2510.16724)의 저장소는 RL 검색 에이전트를 다음 축으로 분류한다. RL의 기능적 역할, cold start 여부, 학습 환경(Real-world vs Simulated), RL 알고리즘, 보상 유형(ORM vs PRM; 규칙 기반 vs LLM 심판), 최적화 범위(single-agent / multi-agent / module-level). — [Awesome-RL-based-Agentic-Search-Papers](https://github.com/ventr1c/Awesome-RL-based-Agentic-Search-Papers)
- 같은 저장소는 2025–2026 대표 서베이로 다음을 나열한다. — [Awesome-RL-based-Agentic-Search-Papers](https://github.com/ventr1c/Awesome-RL-based-Agentic-Search-Papers)
  - "Agentic RAG: A Survey on Agentic RAG"(2501.09136)
  - "Synergizing RAG and Reasoning: A Systematic Review"(2504.15909)
  - "Deep Research Agents: A Systematic Examination and Roadmap"(2506.18096)
  - "A Survey of LLM-based Deep Search Agents"(2508.05668)
  - "Deep Research: A Systematic Survey"(2512.02038)
  - "SoK: Agentic RAG: Taxonomy, Architectures, Evaluation, and Research Directions"(2603.07379)
  - "Agentic Reasoning for LLMs"(2601.12538)
  - "The Landscape of Agentic RL for LLMs: A Survey"(2509.02547, TMLR 표기)
- ACL 2026 Main에는 "A Survey of Large Language Model-Based Search Agents"([2026.acl-long.374](https://aclanthology.org/2026.acl-long.374/))가, ACL 2026 Findings에는 "Data-Centric Perspectives on Agentic RAG: A Survey"([2026.findings-acl.78](https://aclanthology.org/2026.findings-acl.78/))가 채택되었다.

#### 제안하는 하위 범주와 소속 논문 ([n]은 Q1 큐레이션 번호; 학회 라벨은 Q1에서 확인한 것)
- **A. 프롬프트/워크플로 기반(training-free) agentic RAG**
  - [1] Search-o1 (EMNLP 2025 Main)
  - [10] FineVerify (EMNLP 2026 Main)
  - S2G-RAG (ACL 2026 Main)
  - MemSearch-o1 (ACL 2026 Main)
  - Awesome-RAG-Reasoning은 Search-o1을 "Prompting" 범주에 둔다. — [Awesome-RAG-Reasoning](https://github.com/DavidZWZ/Awesome-RAG-Reasoning)
- **B1. RL 학습 검색 에이전트(추론↔검색 교차; 기본 프레임워크)**
  - [2] Search-R1 (COLM 2025)
  - ReSearch (NeurIPS 2025)
  - StepSearch (EMNLP 2025 Main)
  - s3 (EMNLP 2025 Main)
  - AutoRefine (NeurIPS 2025)
  - AceSearcher (NeurIPS 2025 Spotlight)
- **B2. RL 알고리즘·보상 설계(과정 보상 / credit assignment / advantage 보정)**
  - [5] TreePS-RAG (EMNLP 2026 Findings)
  - Tree-GRPO (ICLR 2026)
  - Stratified GRPO (ICML 2026)
  - PiCA (NeurIPS 2026)
  - Dense Process Supervision via Fact Utility (EMNLP 2026 Main)
  - Process vs. Outcome Reward (NeurIPS 2025)
  - CriticSearch (ACL 2026 Findings)
- **B3. 학습 환경·데이터(시뮬레이션 / 합성 과제 / self-play)**
  - [6] SearchGym (ACL 2026 Main)
  - SynSearch (EMNLP 2026 Main)
  - DeepSearch-World (EMNLP 2026 Main)
  - OpenResearcher (EMNLP 2026 Main)
  - Quest (NeurIPS 2026)
  - ZeroSearch / Search Self-Play (학회 Unverified)
- **C. Deep-research / 장기 웹 에이전트**
  - [3] DeepResearcher (EMNLP 2025 Main)
  - [4] DR Tulu (ICML 2026 Spotlight)
  - [9] FS-Researcher (ACL 2026 Main)
  - WebThinker, WebDancer (NeurIPS 2025)
  - RE-TRAC (ICML 2026)
  - Tongyi DeepResearch (Unverified)
- **D. 적응형 검색 / 검색 효율(검색 시점·여부·깊이·중단 결정)**
  - [7] Over-Searching (EACL 2026 Main)
  - [8] AdaSearch (EMNLP 2026 Main)
  - Search Wisely (EMNLP 2025 Main)
  - Adaptive Retrieval for Reasoning / REPAIR (ACL 2026 Main)
  - Knowledge Boundary Aware PO (ACL 2026 Main)
  - AutoSearch, BAPO (ACL 2026 Findings)
  - HALT, One Policy Any Budget (EMNLP 2026 Findings)
  - SpecHop (NeurIPS 2026)
- **E. 멀티에이전트 RAG**
  - [11] JADE (ICML 2026)
  - DECOR (ICML 2026)
  - MHGPO (ACL 2026 Main)
  - MMOA-RAG (NeurIPS 2025)
  - D^2Plan (ACL 2026 Main)
  - DeepDiver-DR (EMNLP 2026 Findings)
- **F. Test-time scaling / inference-time compute for RAG**
  - [10] FineVerify (EMNLP 2026 Main)
  - [9] FS-Researcher (ACL 2026 Main)
  - CoRAG (NeurIPS 2025)
  - BrowseConf (ACL 2026 Findings)
  - Beyond Confidence (EMNLP 2026 Findings)
- **G. (신흥) 에이전트용 검색 인터페이스 재설계**
  - [12] DCI (NeurIPS 2026 Spotlight)
  - Retriever-Free Corpus-Traversing MCTS (NeurIPS 2026)
  - Agentic-R (ACL 2026 Findings)
  - DCI 후속 연구(모두 학회 Unverified): GrepSeek ([2605.29307](https://arxiv.org/abs/2605.29307)), Dr-DCI ([2606.14885](https://arxiv.org/abs/2606.14885)), "A New Role for Relevance: Guiding Corpus Interaction in Agentic Search" ([2607.24223](https://arxiv.org/abs/2607.24223)), "Towards Retrieving Interaction Spaces for Agentic Search" ([2606.06880](https://arxiv.org/abs/2606.06880))

### Inferences
- 서베이의 "Prompting / SFT / RL" 축과 "single / multi-agent" 축만으로는 2026년 연구를 충분히 구분하기 어렵다. 2026년 논문의 다수가 RL 학습 에이전트이기 때문이다. 그래서 B 범주를 알고리즘·보상(B2)과 환경·데이터(B3)로 나누는 것이 현재 문헌 분포에 더 맞다.
- C(deep research)와 F(test-time scaling)는 점점 겹치고 있다. FS-Researcher처럼 계산량 배분 자체를 기여로 내세우는 논문이 늘고 있어, 보고서에서는 교차 범주로 표시하는 것이 좋다.
- G는 "리트리버" 연구(다른 조사자 범위)와 경계에 있다. 다만 DCI는 에이전트가 검색 도구를 어떻게 쓰는가를 다루므로 agentic RAG 쪽에 두되, 리트리버 담당자와 중복 확인이 필요하다.

### Gaps
- 2026년 SoK(2603.07379)와 "Agentic Reasoning for LLMs"(2601.12538) 서베이의 원문 분류 체계는 arXiv에 접속할 수 없어 직접 읽지 못했다. 위 분류는 두 awesome 저장소에 반영된 서베이 분류와 학회 채택 논문 분포를 바탕으로 한 제안이다.
- 학회 채택 논문 외에 각 하위 범주의 정량적 비중(논문 수)을 정확히 세지는 않았다(Q3의 키워드 카운트 참조).

## Q3. 2026년에 떠오르는 최신 방향은 무엇인가? (보상 설계, 과정 보상, 검색 호출 효율, 시뮬레이션 검색 환경 등)

### Takeaway
학회 채택 논문 수가 급증했다. 제목 키워드 기준으로 EMNLP Main에서 2025년 7편이 2026년 38편이 되었다. 2026년 새 방향은 7가지로 정리된다.
- (1) 장문 deep research용 rubric 보상
- (2) 과정 보상과 세밀한 credit assignment(트리 롤아웃, 정보 이득, 사실 효용)
- (3) 과잉 검색 억제와 적응형·예산 인지 검색
- (4) 시뮬레이션·합성 검색 환경
- (5) 장기 컨텍스트·메모리 관리
- (6) agentic search의 test-time scaling과 자기검증
- (7) 검색 인터페이스와 에이전트의 공동 설계

### Cited Findings
- **정량 신호(제목 키워드 매칭)**: 정규식 `search agent|agentic search|deep research|deep search|agentic rag|agentic retrieval|searcher|search-r1|search-augmented`로 학회 채택 논문 제목을 셌다. 평가·안전 논문도 포함되므로 대략적인 추세 지표다.
  - NeurIPS 2025: 8/6212 → ICML 2026: 19/6341 → NeurIPS 2026: 27/6437 — [papercopilot/paperlists](https://github.com/papercopilot/paperlists)
  - EMNLP 2025 Main: 7/1809 → ACL 2026 Main(long): 31/2222 → EMNLP 2026 Main: 38/2704 — [acl-org/acl-anthology](https://github.com/acl-org/acl-anthology)
  - EMNLP 2025 Findings: 4/1405 → ACL 2026 Findings: 18/2163 → EMNLP 2026 Findings: 35/2478 — [acl-org/acl-anthology](https://github.com/acl-org/acl-anthology)
- **(1) 장문 rubric 보상**
  - DR Tulu(ICML 2026 Spotlight)는 RLVR이 장문 과제로 확장되지 않는 문제를, 정책과 공진화하는 루브릭으로 해결한다. — [OpenReview](https://openreview.net/forum?id=97NEP1pyS3)
  - 후속 채택작: "Chaining the Evidence: … Citation-Aware Rubric Rewards"(ACL 2026 Main, [2026.acl-long.950](https://aclanthology.org/2026.acl-long.950/)), DEEPRUBRIC(EMNLP 2026 Main, [2026.emnlp-main.2267](https://aclanthology.org/2026.emnlp-main.2267/)), "Deep Research as Rubric"(NeurIPS 2026, [JmP1iLzzlJ](https://openreview.net/forum?id=JmP1iLzzlJ)), LongTraceRL(EMNLP 2026 Findings, [2026.findings-emnlp.1222](https://aclanthology.org/2026.findings-emnlp.1222/))
  - 웹 검색 결과에서 DR-Rubric, DeepRubric, HOTE가 DR Tulu를 기반으로 확장한 후속 연구로 확인되었다. — [arXiv 2606.17029](https://arxiv.org/abs/2606.17029), [arXiv 2606.13710](https://arxiv.org/abs/2606.13710)
- **(2) 과정 보상 / credit assignment**
  - TreePS-RAG: 롤아웃 트리와 몬테카를로 단계 효용 — [2026.findings-emnlp.1007](https://aclanthology.org/2026.findings-emnlp.1007/)
  - Tree-GRPO(ICLR 2026): 트리 탐색 기반 그룹 RL로, outcome 보상만으로도 단계별 과정 신호를 얻는다. — [iclr.cc](https://iclr.cc/virtual/2026/poster/10008772)
  - PiCA(NeurIPS 2026): 누적 정보 이득에서 "pivot" 단계를 찾고 PBRS 기반으로 보상한다. 3B/7B 모델에서 15.2%/2.2% 향상. — [OpenReview](https://openreview.net/forum?id=4CNCefRkpK)
  - Stratified GRPO(ICML 2026): 도구 호출 구조별로 계층화한 advantage 정규화로 GRPO 대비 최대 12.6점 향상. — [OpenReview](https://openreview.net/forum?id=ZX2lOgKOTm)
  - Dense Process Supervision via Fact Utility(EMNLP 2026 Main) — [2026.emnlp-main.1050](https://aclanthology.org/2026.emnlp-main.1050/)
  - 2025년 NeurIPS의 "Process vs. Outcome Reward" 논문이 이 흐름의 출발점에 있다. — [papercopilot/paperlists](https://github.com/papercopilot/paperlists)
- **(3) 과잉 검색 억제 / 적응형·예산 인지 검색**
  - Over-Searching(EACL 2026 Main): 과잉 검색이 복잡 추론 모델과 deep research 시스템에서 더 심하다는 실증과 TPC 지표 — [2026.eacl-long.361](https://aclanthology.org/2026.eacl-long.361/)
  - AdaSearch(EMNLP 2026 Main): 검색 결정을 문제 해결과 분리 — [2026.emnlp-main.2696](https://aclanthology.org/2026.emnlp-main.2696/)
  - 관련 채택작: AutoSearch "Adaptive Search Depth"(ACL 2026 Findings, [2026.findings-acl.1399](https://aclanthology.org/2026.findings-acl.1399/)), HALT "Verification-Aware Stopping"([2026.findings-emnlp.343](https://aclanthology.org/2026.findings-emnlp.343/)), "One Policy, Any Budget: Internalizing Budget-Aware Search via RL"([2026.findings-emnlp.1857](https://aclanthology.org/2026.findings-emnlp.1857/)), "Not Worth Another Token: Marginal Value Estimation for Efficient Deep Research Agents"([2026.findings-emnlp.2218](https://aclanthology.org/2026.findings-emnlp.2218/)), "To Answer or to Abstain"(EMNLP 2026 Main, [2026.emnlp-main.161](https://aclanthology.org/2026.emnlp-main.161/))
- **(4) 시뮬레이션·합성 검색 환경**
  - SearchGym(ACL 2026 Main)은 지식그래프와 정렬 코퍼스로 "strictly solvable" 과제를 만들고, 웹 기반 ASearcher보다 평균 상대 10.6% 높다. — [2026.acl-long.848](https://aclanthology.org/2026.acl-long.848/)
  - SynSearch(EMNLP 2026 Main)는 환경과 과제 분포를 공동 설계한 완전 합성 search–browse 환경으로, 실제 검색 평가에서도 강한 성능을 보인다. — [2026.emnlp-main.928](https://aclanthology.org/2026.emnlp-main.928/)
  - 관련 채택작: DeepSearch-World(EMNLP 2026 Main, [2026.emnlp-main.1117](https://aclanthology.org/2026.emnlp-main.1117/)), Quest(NeurIPS 2026, [eKFW4EIOJx](https://openreview.net/forum?id=eKFW4EIOJx)), OffSeeker "Online RL Is Not All You Need for Deep Research Agents"(EMNLP 2026 Main, [2026.emnlp-main.173](https://aclanthology.org/2026.emnlp-main.173/))
- **(5) 장기 컨텍스트·메모리 관리**
  - FS-Researcher(ACL 2026 Main): 파일 시스템을 외부 메모리로 사용 — [2026.acl-long.288](https://aclanthology.org/2026.acl-long.288/)
  - MemSearch-o1(ACL 2026 Main): think–search 루프에서 생기는 "memory dilution"을 다룬다. — [2026.acl-long.41](https://aclanthology.org/2026.acl-long.41/)
  - RE-TRAC(ICML 2026, [wjNTbZKIvC](https://openreview.net/forum?id=wjNTbZKIvC)), ARC(ACL 2026 Findings, [2026.findings-acl.930](https://aclanthology.org/2026.findings-acl.930/)), "Beyond Turn Limits: … Dynamic Context Window"([2026.findings-emnlp.2461](https://aclanthology.org/2026.findings-emnlp.2461/))
- **(6) agentic search의 test-time scaling / 자기검증**
  - FineVerify(EMNLP 2026 Main): 샘플 12개로 GPT-5-mini가 BrowseComp-Plus에서 GPT-5를 넘어섰다. — [2026.emnlp-main.2599](https://aclanthology.org/2026.emnlp-main.2599/)
  - BrowseConf(ACL 2026 Findings, [2026.findings-acl.21](https://aclanthology.org/2026.findings-acl.21/)), "Inference-Time Scaling of Verification: Self-Evolving Deep Research Agents via Test-Time Rubric-Guided Verification"(ACL 2026 Findings, [2026.findings-acl.1243](https://aclanthology.org/2026.findings-acl.1243/)), CLIFT(NeurIPS 2026, [igVA4JNEyQ](https://openreview.net/forum?id=igVA4JNEyQ))
- **(7) 검색 인터페이스·리트리버와 에이전트의 공동 설계**
  - DCI(NeurIPS 2026 Spotlight): grep/셸로 코퍼스를 직접 탐색하는 방식이 sparse/dense/rerank 베이스라인을 앞선다. — [OpenReview](https://openreview.net/forum?id=42dIxEhArH)
  - 웹 검색 결과에 따르면, 후속 연구 "Towards Retrieving Interaction Spaces"는 광범위한 셸 명령이 매번 전체 코퍼스를 스캔하므로 무제한 상호작용은 확장되지 않는다는 한계를 지적한다. — [arXiv 2606.06880](https://arxiv.org/abs/2606.06880)
  - CoSearch(2604.17555)는 추론 에이전트와 생성형 문서 랭커를 GRPO로 공동 학습하며, Search-R1류가 검색 시스템을 고정 도구로 취급한다는 점을 문제로 든다. 학회는 Unverified. — [arXiv 2604.17555](https://arxiv.org/abs/2604.17555)
- **(8) 멀티에이전트 공동 RL** (보조 방향)
  - JADE(ICML 2026), DECOR(ICML 2026; Planner/Filter/Answerer 역할을 MARL로 공동 최적화), MHGPO(ACL 2026 Main; critic 없는 이종 그룹 상대 advantage) — [OpenReview dkRAVqr347](https://openreview.net/forum?id=dkRAVqr347), [2026.acl-long.1399](https://aclanthology.org/2026.acl-long.1399/)

### Inferences
- 2025년의 핵심 질문은 "RL로 검색을 학습시킬 수 있는가"(Search-R1)였다. 2026년에는 "무엇을 보상하고(rubric·과정 보상), 어디서 학습하고(시뮬레이션), 얼마나 검색할 것인가(과잉 검색·예산)"로 옮겨 갔다. outcome-only 보상의 한계와 실제 웹 API의 비용·비재현성이 공통 동기다.
- 학회 채택 수로 보면 2026년 ACL·EMNLP의 Main과 Findings 모두에서 이 하위 분야가 크게 늘었다. 키워드 카운트는 대략적이지만 증가 폭이 몇 배에 이르므로 추세 판단에는 충분하다고 본다.
- "검색 인터페이스 재설계"(DCI)와 "리트리버-에이전트 공동 학습"(CoSearch, Agentic-R)은 2026년 후반의 새 흐름으로 보인다. 다만 대부분 아직 프리프린트이거나 1편만 채택된 상태라, 정착된 방향이라기보다 초기 신호로 다루는 것이 안전하다.

### Gaps
- ICLR 2027 제출작(심사 중, 결정 없음)과 2026년 9–11월 arXiv 신작은 학회 검증이 불가능하므로 체계적으로 조사하지 않았다. 웹 검색에서는 "Contextual Information Policy Optimization for Search Agents"(arXiv [2608.06128](https://arxiv.org/abs/2608.06128)), "SOAR: Source-Aware RL with Precise Credit Assignment for Search Agents"(arXiv [2610.13157](https://arxiv.org/abs/2610.13157)) 같은 최신 프리프린트가 보였다. 모두 Preprint이며 초록은 확인하지 못했다.
- 키워드 카운트에는 평가·안전·멀티모달 논문(범위 밖)이 섞여 있다. 하위 범주별 정밀 집계는 하지 않았다.
- AAAI 2026, WWW 2026, SIGIR 2026, KDD 2026의 채택 논문은 공식 목록에 접근하지 못해 추세 집계에서 빠졌다.
