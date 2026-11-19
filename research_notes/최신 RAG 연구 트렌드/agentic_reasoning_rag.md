# Agentic RAG & 추론 통합 검색(Reasoning-Integrated Retrieval) — 2026-11-19 기준 연구 노트 (1차 출처 재검증판)

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

아래 12편은 모두 arXiv abs 페이지에서 존재 여부, 제목, 1저자, v1 일자, 초록을 직접 확인했다. 학회 채택 여부는 다음으로 확인했다.
- 8편: ACL Anthology 페이지 또는 arXiv Comments
- 4편(Search-R1, DR Tulu의 Spotlight 표기, JADE, DCI): OpenReview 결정을 미러링한 Paper Copilot 데이터. OpenReview 원문은 이 세션에서 열 수 없었다(아래 참조).

### Cited Findings

#### 검증 방법과 한계 (먼저 읽을 것)
- **arXiv (직접 확인 ✅)**: 12편 전부와 본문에 인용된 나머지 arXiv ID 30개를 `arxiv.org/abs/<id>` 페이지에서 직접 받아 확인했다. 확인 항목은 제목, 저자, Submission history(v1 일시), Comments, 초록이다. `export.arxiv.org` API는 "Rate exceeded"로 응답해 abs 페이지를 사용했다. — [arXiv](https://arxiv.org/)
- **ACL Anthology (직접 확인 ✅)**: 본문에 인용된 Anthology ID 53개 전부를 `aclanthology.org/<id>/` 페이지에서 직접 받았다. `citation_title`, `citation_author`, `citation_conference_title` 메타데이터로 제목, 1저자, Main/Findings 여부를 확인했다. — [ACL Anthology](https://aclanthology.org/)
- **OpenReview (직접 확인 불가 ⚠️)**: 이번에도 열지 못했다.
  - `api.openreview.net`와 `openreview.net/forum`은 HTTP 403 "ChallengeRequiredError: Challenge verification required"(봇 검증)로 응답했다.
  - `api2.openreview.net`은 프록시가 차단했고, WebFetch는 `openreview.net`의 DNS 조회에 실패했다.
  - icml.cc, neurips.cc, iclr.cc, colmweb.org, dblp.org, proceedings.mlr.press, proceedings.neurips.cc도 모두 프록시 403이었다.
  - 그래서 COLM/ICML/NeurIPS/ICLR 결정은 OpenReview 미러인 Paper Copilot JSON(`raw.githubusercontent.com` / `media.githubusercontent.com`)의 `status` 필드로 다시 확인했다. 미러 항목에는 OpenReview forum ID와 OpenReview 초록이 함께 들어 있다. 이번에는 Git LFS였던 ICLR 2026 파일도 받았다. — [papercopilot/paperlists](https://github.com/papercopilot/paperlists)
  - arXiv Comments에 학회명이 있으면 그것을 1차 근거로 삼았다(DR Tulu, Tree-GRPO, Search Self-Play, SIGIR 2026 두 편 등).
- 초록 버전 차이: ACL Anthology(학회판) 초록과 arXiv 최신판 초록은 문구와 수치가 다른 경우가 있다(Search-o1, DeepResearcher, DCI, Stratified GRPO). 아래 각 항목에 어느 판의 문구인지 표시했다.

#### 큐레이션 목록 (12편; [F] = 2025년 초 기초(foundational) 논문)

**[1][F] Search-o1: Agentic Search-Enhanced Large Reasoning Models** — Xiaoxi Li et al. (8인)
- arXiv: [2501.05366](https://arxiv.org/abs/2501.05366) | 최초 공개: **2025-01-09** (v1, 단일 버전)
- 학회: **EMNLP 2025 Main** — [2025.emnlp-main.276](https://aclanthology.org/2025.emnlp-main.276/) ("Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing")
- 요약 1: o1류 대형 추론 모델(LRM)은 긴 추론 도중 지식 부족으로 불확실성과 오류를 자주 겪는다. 이를 해결하려고, 불확실한 지식 지점에서 외부 지식을 동적으로 검색하는 agentic 검색 워크플로를 추론 과정에 통합한다. 장황한 검색 문서는 별도의 Reason-in-Documents 모듈이 깊이 분석해 잡음을 줄인 뒤 추론 체인에 주입한다.
- 요약 2: 과학·수학·코딩 복잡 추론 과제와 6개 오픈도메인 QA 벤치마크에서 강한 성능을 보였다. 복잡 추론에서 LRM의 신뢰성을 높인다. 학회판 초록은 이를 "고급 deep research 시스템으로 가는 길", arXiv판 초록은 "더 신뢰할 수 있고 범용적인 지능형 시스템으로 가는 길"로 표현한다.
- 인용: "integrates an agentic search workflow into the reasoning process" (arXiv·학회판 공통)
- 하위분류: 프롬프트/워크플로 기반 agentic RAG. Awesome-RAG-Reasoning도 "Single-Agent › Prompting"으로 분류한다. — [Awesome-RAG-Reasoning](https://github.com/DavidZWZ/Awesome-RAG-Reasoning)
- 검증: ✅ arXiv abs 확인(제목·1저자·v1 2025-01-09·초록 일치; Comments 없음) / venue 출처: ACL Anthology 페이지 메타데이터(EMNLP 2025 Main 볼륨)

**[2][F] Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning** — Bowen Jin et al. (8인)
- arXiv: [2503.09516](https://arxiv.org/abs/2503.09516) | 최초 공개: **2025-03-12** (v1; v5 2025-08-05까지 개정)
- 학회: **COLM 2025 (Poster)** — [OpenReview Rwhi91ideu](https://openreview.net/forum?id=Rwhi91ideu) (Paper Copilot `colm2025.json` status = Poster)
- 요약 1: 추론 능력이 있는 LLM에게 프롬프트만으로 검색엔진을 쓰게 하면, 검색엔진과 최적으로 상호작용하는 능력이 없어 결과가 최적에 못 미친다. 이를 해결하려고, LLM이 단계별 추론 중에 실시간 검색과 함께 (여러 개의) 검색 쿼리를 스스로 생성하도록 RL로 학습시킨다. 안정적인 학습을 위해 검색된 토큰을 마스킹하고, 단순한 결과(outcome) 기반 보상을 쓴다.
- 요약 2: 7개 QA 데이터셋에서 같은 설정의 여러 RAG 베이스라인보다 Qwen2.5-7B는 41%, Qwen2.5-3B는 20% 향상했다(arXiv 최신판 초록 수치). RL 최적화 방법, LLM 선택, 응답 길이 동역학에 대한 경험적 통찰도 함께 제시한다.
- 인용: "leveraging retrieved token masking for stable RL training and a simple outcome-based reward function"
- 하위분류: RL 학습 검색 에이전트. 이후 대부분의 2025–2026 연구가 비교 기준으로 사용한다.
- 검증: ✅ arXiv abs 확인(제목·1저자·v1·초록 일치; Comments "31 pages"로 학회 표기 없음) / ⚠️ venue 출처: Paper Copilot OpenReview 미러(status=Poster). OpenReview 원문은 봇 검증 403으로 직접 열람 불가.

**[3][F] DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments** — Yuxiang Zheng et al. (7인)
- arXiv: [2504.03160](https://arxiv.org/abs/2504.03160) | 최초 공개: **2025-04-04** (v1; v4 2025-04-17)
- 학회: **EMNLP 2025 Main** — [2025.emnlp-main.22](https://aclanthology.org/2025.emnlp-main.22/)
- 요약 1: 기존 deep research 방법(취약한 프롬프트 엔지니어링, 통제된 RAG 환경에서의 RL)은 현실 웹 상호작용의 복잡성을 담지 못한다. 이를 해결하려고, 실제 웹 검색 상호작용 환경에서 RL을 스케일링해 LLM 기반 deep research 에이전트를 end-to-end로 학습하는 첫 종합 프레임워크를 제안한다. 다양한 웹페이지 구조에서 정보를 추출하는 브라우징 에이전트를 둔 특화 멀티에이전트 구조로, 잡음 많고 비정형적이며 동적인 오픈 웹을 다루도록 학습한다.
- 요약 2: 오픈도메인 리서치 과제에서 프롬프트 엔지니어링 기반 대비 최대 28.9점, RAG 기반 RL 에이전트 대비 최대 7.2점 향상했다. 계획 수립, 다중 출처 교차검증, 자기성찰을 통한 방향 전환, 확답할 수 없을 때 정직하게 답하는 행동이 end-to-end RL 학습에서 창발했다.
- 인용(학회판 초록): "end-to-end training in real-world web environments is fundamental for developing robust research capabilities". arXiv판 문구는 "…is not merely an implementation detail but a fundamental requirement for developing robust research capabilities"다.
- 하위분류: Deep-research / 장기 웹 에이전트 (실제 웹 환경 RL)
- 검증: ✅ arXiv abs 확인(제목·1저자·v1·초록 일치; Comments 없음) / venue 출처: ACL Anthology 페이지(EMNLP 2025 Main 볼륨)

**[4] DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research** — Rulin Shao et al. (21인)
- arXiv: [2511.19399](https://arxiv.org/abs/2511.19399) | 최초 공개: **2025-11-24** (v1; v3 2026-05-15)
- 학회: **ICML 2026 (Spotlight)** — arXiv Comments "ICML 2026". Spotlight 등급은 [OpenReview 97NEP1pyS3](https://openreview.net/forum?id=97NEP1pyS3) 미러의 status = Spotlight로 확인했다.
- 요약 1: 대부분의 오픈 deep research 에이전트는 검증하기 쉬운 단답형 QA로 RLVR 학습을 받기 때문에, 현실적인 장문 과제로 확장되지 않는다. 이를 해결하려고, 학습 중에 루브릭을 구성·유지하며 정책 모델과 공진화(co-evolve)시키는 RLER(Reinforcement Learning with Evolving Rubrics)를 제안한다. 루브릭이 검색으로 새로 탐색한 정보와 대비되는 모델 응답을 반영하므로, 사실 확인과 on-policy 피드백의 변별력이 좋아진다. 이를 이용해 DR Tulu-8B를 개발했다.
- 요약 2: 과학·헬스케어·일반 도메인의 장문 deep research 벤치마크 4개에서 기존 오픈 에이전트를 크게 앞섰고(Tongyi DR 대비 평균 +15.6%), 독점 에이전트와 같거나 더 높았다(OpenAI DR 대비 평균 +0.7%). 모델은 훨씬 작고, 쿼리당 비용은 OpenAI DR보다 1000배 저렴하다. 개방형 장문 deep research용으로 직접 학습한 첫 완전 오픈 모델이다.
- 인용: "rubrics are constructed and maintained to co-evolve with the policy model during training"
- 하위분류: Deep-research 에이전트 / 보상 설계(rubric reward)
- 검증: ✅ arXiv abs 확인(제목·1저자·v1·초록 일치) / venue 출처: arXiv Comments "ICML 2026"(1차). ⚠️ Spotlight 등급은 Paper Copilot OpenReview 미러로만 확인.

**[5] TreePS-RAG: Tree-based Process Supervision for Reinforcement Learning in Agentic RAG** — Tianhua Zhang et al. (8인)
- arXiv: [2601.06922](https://arxiv.org/abs/2601.06922) | 최초 공개: **2026-01-11** (v1, 단일 버전)
- 학회: **EMNLP 2026 Findings** — [2026.findings-emnlp.1007](https://aclanthology.org/2026.findings-emnlp.1007/) ("Findings of the Association for Computational Linguistics: EMNLP 2026")
- 요약 1: 결과 기반의 희소한 최종 보상만으로는 agentic RAG의 단계별 credit assignment가 약하다. 기존 과정 감독은 오프라인으로 만든 데이터(분포 이동 위험)나 비싼 중간 주석에 의존한다. 이를 해결하려고, agentic RAG 추론을 각 단계가 노드인 롤아웃 트리로 모델링하는 온라인 트리 기반 RL을 제안한다. 각 단계의 효용은 후손 노드 결과에 대한 몬테카를로 추정으로 구하고, 보상은 표준 결과 보상만 유지한다. 계산 예산 안에서 탐색 다양성을 지키는 효율적인 온라인 트리 구성 전략도 함께 제안한다.
- 요약 2: Search-R1 같은 강한 베이스라인과 비슷한 롤아웃 비용으로, 7개 멀티홉·일반 QA 벤치마크와 여러 모델 규모에서 결과 감독 RL과 기존 과정 감독 RL 방법을 일관되고 유의하게 앞선다.
- 인용: "model agentic RAG reasoning as a rollout tree, where each reasoning step naturally maps to a node"
- 하위분류: RL 검색 에이전트 › 과정 보상 / credit assignment
- 검증: ✅ arXiv abs 확인(제목·1저자·v1·초록 일치; Comments 없음) / venue 출처: ACL Anthology 페이지(EMNLP 2026 Findings 볼륨)

**[6] SearchGym: Bootstrapping Real-World Search Agents via Cost-Effective and High-Fidelity Environment Simulation** — Xichen Zhang et al. (9인)
- arXiv: [2601.14615](https://arxiv.org/abs/2601.14615) | 최초 공개: **2026-01-21** (v1, 단일 버전)
- 학회: **ACL 2026 Main** (Long) — [2026.acl-long.848](https://aclanthology.org/2026.acl-long.848/) ("64th Annual Meeting of the ACL (Volume 1: Long Papers)", DOI 10.18653/v1/2026.acl-long.848)
- 요약 1: 검색 에이전트 RL 학습에는 딜레마가 있다. 상용 웹 API와 상호작용하면 비용이 지나치게 크고, 정적 데이터 스냅샷을 쓰면 데이터 불일치로 보상 신호가 오염된다. 이를 해결하려고, 검증 가능한 지식그래프와 그에 정렬된 문서 코퍼스를 생성해 모든 과제가 사실에 근거하고 반드시 풀 수 있도록 만든 시뮬레이션 환경 SearchGym을 제안한다. 여기에 기본 상호작용에서 장기 계획으로 점진적으로 나아가는 커리큘럼 학습법 SearchGym-RL을 더했다.
- 요약 2: Llama·Qwen 계열에서 강한 Sim-to-Real 일반화를 보였다. SearchGym에서 학습한 Qwen2.5-7B-Base는 9개 벤치마크에서 웹을 쓰는 ASearcher 베이스라인보다 평균 상대 10.6% 높다.
- 인용: "high-fidelity simulation serves as a scalable and highly cost-effective methodology"
- 하위분류: RL 검색 에이전트 › 시뮬레이션 학습 환경 (ZeroSearch 계열의 후속 방향)
- 검증: ✅ arXiv abs 확인(제목·1저자·v1·초록 일치; Comments 없음) / venue 출처: ACL Anthology 페이지(ACL 2026 Long 볼륨)

**[7] Over-Searching in Search-Augmented Large Language Models** — Roy Xie et al. (7인)
- arXiv: [2601.05503](https://arxiv.org/abs/2601.05503) | 최초 공개: **2026-01-09** (v1; v2 2026-03-11)
- 학회: **EACL 2026 Main** (Long) — [2026.eacl-long.361](https://aclanthology.org/2026.eacl-long.361/). arXiv Comments도 "Accepted to EACL 2026 Main Conference"다.
- 요약 1: 검색 증강 LLM은 응답 품질이 나아지지 않는데도 검색 도구를 불필요하게 호출하는 과잉 검색(over-searching)을 보인다. 이는 계산 비효율과, 무관한 컨텍스트 유입에 따른 환각을 낳는다. 이 현상을 질의 유형, 모델 범주, 검색 조건, 다중 턴 대화의 여러 차원에서 체계적으로 평가했다.
- 요약 2: 주요 발견은 세 가지다. (i) 검색은 답할 수 있는 질의의 정확도는 높이지만, 답할 수 없는 질의에서는 기권(abstention)을 해친다. (ii) 과잉 검색은 복잡한 추론 모델과 deep research 시스템에서 더 심하고, 잡음 섞인 검색에서 악화되며, 다중 턴에서 누적된다. (iii) 부정 증거(negative evidence)가 있으면 기권이 개선된다. 비용과 성능의 트레이드오프를 재는 지표 TPC(Tokens Per Correctness)를 제안했고, 질의·검색 수준의 완화책을 검토했으며, OverSearchQA를 공개했다.
- 인용: "unnecessarily invoking search tool even when it does not improve response quality"
- 하위분류: 적응형 검색 / 검색 호출 효율
- 검증: ✅ arXiv abs 확인(제목·1저자·v1·초록 일치) / venue 출처: ACL Anthology 페이지(EACL 2026 Long 볼륨)와 arXiv Comments

**[8] AdaSearch: Balancing Parametric Knowledge and Search in Large Language Models via Reinforcement Learning** — Tzu-Han Lin et al. (6인)
- arXiv: [2512.16883](https://arxiv.org/abs/2512.16883) | 최초 공개: **2025-12-18** (v1; v2 2026-09-01)
- 학회: **EMNLP 2026 Main** — [2026.emnlp-main.2696](https://aclanthology.org/2026.emnlp-main.2696/). arXiv Comments도 "Accepted to EMNLP 2026"이다.
- 요약 1: RL 검색 에이전트가 내부 파라메트릭 지식과 외부 검색 사이의 균형을 적응적으로 잡지 못하는 문제를 다룬다. 검색에 지나치게 의존하면 비용이 늘고 잡음·악성 콘텐츠에 노출되며, 파라메트릭 지식에만 의존하면 환각이 생긴다. 기존의 도구 호출 보상 셰이핑은 보상 설계 부담이 크고, 필요한 검색과 불필요한 검색을 구분하지 못한다. 저자들은 F1 기반 결정 지표로 평가를 다시 해 기존 방법이 이미 가진 파라메트릭 지식을 간과한다는 점을 보였다. 이를 바탕으로 문제 해결과 검색 여부 결정을 분리한 단순한 2단계 결과 기반 RL 프레임워크 AdaSearch를 제안한다.
- 요약 2: 검색 결정 품질을 크게 높이고 불필요한 검색 호출을 줄였다. QA 정확도는 항상 검색하는 방식보다 약간 낮아지는 정도에 그쳤고, 결정 과정을 명시적이고 해석 가능하게 만들었다.
- 인용: "disentangles problem-solving from the decision to search"
- 하위분류: 적응형 검색(검색 여부 결정) × RL
- 검증: ✅ arXiv abs 확인(이전 노트의 "웹 검색 요약에서 얻은 ID" 문제 해소; 제목·1저자·초록 일치) / venue 출처: ACL Anthology 페이지(EMNLP 2026 Main 볼륨)와 arXiv Comments

**[9] FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents** — Chiwei Zhu et al. (7인)
- arXiv: [2602.01566](https://arxiv.org/abs/2602.01566) | 최초 공개: **2026-02-02** (v1; v2 2026-04-17)
- 학회: **ACL 2026 Main** (Long) — [2026.acl-long.288](https://aclanthology.org/2026.acl-long.288/). arXiv Comments도 "Accepted to ACL 2026"이다.
- 요약 1: deep research의 긴 궤적은 컨텍스트 한계를 넘기 쉬워, 증거 수집과 보고서 작성의 토큰 예산을 압박하고 test-time scaling을 가로막는다. 이를 해결하려고, 파일 시스템을 지속적 작업공간으로 쓰는 이중 에이전트 프레임워크를 제안한다. 사서 역할의 Context Builder가 웹을 탐색하며 구조화된 노트를 쓰고 원문을 계층적 지식베이스에 보관하면, Report Writer가 이를 사실 출처로 삼아 섹션별로 보고서를 작성한다.
- 요약 2: DeepResearch Bench와 DeepConsult 두 개방형 벤치마크에서 여러 백본 모델로 SOTA 보고서 품질을 달성했다. Context Builder에 배정한 계산량과 최종 보고서 품질이 양의 상관을 보여, 파일 시스템 패러다임에서도 test-time scaling이 작동함을 입증했다.
- 인용: "the file system serves as a durable external memory and a shared coordination medium across agents and sessions"
- 하위분류: Deep-research 에이전트 × test-time scaling × 멀티에이전트
- 검증: ✅ arXiv abs 확인(제목·1저자·v1·초록 일치) / venue 출처: ACL Anthology 페이지(ACL 2026 Long 볼륨)와 arXiv Comments

**[10] FineVerify: Scaling Test-Time Compute with Fine-Grained Self-Verification for Agentic Search** — James Xu Zhao et al. (4인)
- arXiv: [2606.00660](https://arxiv.org/abs/2606.00660) | 최초 공개: **2026-05-30** (v1; v2 2026-09-01). ID는 2606이지만 v1 일자는 5월 30일(UTC)이다.
- 학회: **EMNLP 2026 Main** — [2026.emnlp-main.2599](https://aclanthology.org/2026.emnlp-main.2599/). arXiv Comments도 "EMNLP 2026 main"이다.
- 요약 1: agentic search에서 test-time compute를 늘려도 실패할 수 있다. 정답이 드물고, 점수 기반 선택이 모델의 보정(calibration)에 의존하기 때문이다. 이를 해결하려고, 질문을 검증 가능한 하위 질문으로 분해하고, 샘플링한 후보 답을 하위 질문별로 검증한 뒤, 집계 점수가 가장 높은 후보를 고르는 세밀한 자기검증 프레임워크를 제안한다. 같은 명시적 기준에 따른 국소 판단으로 선택 문제를 단순화한다.
- 요약 2: 4개 agentic search 벤치마크와 2개 모델에서 표준 스케일링 기법을 일관되게 앞섰다. 궤적 4개만으로 GPT-5-mini는 평균 8.2점, Gemini-3-flash는 평균 5.6% 향상했고, 샘플 12개로 GPT-5-mini가 BrowseComp-Plus에서 GPT-5를 넘어섰다. 해석 가능한 검증 기록은 벤치마크 오류 감사에도 쓸 수 있다.
- 인용: "decomposes each question into checkable sub-questions"
- 하위분류: 프롬프트/추론 시점 기법 × test-time scaling (학습 불필요)
- 검증: ✅ arXiv abs 확인(제목·1저자·v1·초록 일치) / venue 출처: ACL Anthology 페이지(EMNLP 2026 Main 볼륨)와 arXiv Comments

**[11] JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG** — Yiqun Chen et al. (11인)
- arXiv: [2601.21916](https://arxiv.org/abs/2601.21916) | 최초 공개: **2026-01-29** (v1, 단일 버전)
- 학회: **ICML 2026 (Poster)** — [OpenReview KQT6pf4Mva](https://openreview.net/forum?id=KQT6pf4Mva) (Paper Copilot `icml2026.json` status = Poster). 같은 학회에 동명 논문 "JADE: Expert-Grounded Dynamic Evaluation…"(SoilRyCv1i)이 있으니 혼동하지 말 것.
- 요약 1: 동적 agentic RAG의 기존 방식은 둘 중 하나다. 고정 그래프 구조 안에서 모듈을 함께 최적화하거나, 동적으로 계획하되 실행기(executor)를 동결된 블랙박스 도구로 둔다. 이렇게 최적화가 분리되면 정교한 계획이 적응되지 않은 실행기 때문에 실현되지 못하는 "전략-운영 불일치"가 생기고, 시스템이 복잡해졌는데도 성능이 오히려 떨어지기도 한다. 이를 해결하려고, 단일 공유 백본 위의 협력 멀티에이전트 팀으로 계획과 실행을 결과 보상 기반으로 end-to-end 공동 최적화하는 JADE(Joint Agentic Dynamic Execution)를 제안한다.
- 요약 2: planner는 executor의 능력 범위 안에서 계획하고, executor는 상위 전략 의도에 맞게 진화하는 co-adaptation이 일어난다. 이를 통해 분리되어 있던 모듈을 시너지를 내는 시스템으로 바꿔 성능을 크게 높이고, 동적 워크플로 오케스트레이션으로 효율과 효과의 균형을 유연하게 조절한다. arXiv 초록에는 구체적 수치가 없다.
- 인용(arXiv 원문): "the planner learns to operate within the capability boundaries of the executors"
- 하위분류: 멀티에이전트 RAG × RL 공동 최적화
- 검증: ✅ arXiv abs 확인(제목·1저자·v1·초록 일치; Comments 없음) / ⚠️ venue 출처: Paper Copilot OpenReview 미러만(status=Poster). OpenReview 원문은 봇 검증 403, arXiv Comments에는 학회 표기가 없다.

**[12] Beyond Semantic Similarity: Rethinking Retrieval for Agentic Search via Direct Corpus Interaction** — Zhuofeng Li et al. (19인)
- arXiv: [2605.05242](https://arxiv.org/abs/2605.05242) | 최초 공개: **2026-05-03** (v1, 단일 버전)
- 학회: **NeurIPS 2026 (Spotlight)** — [OpenReview 42dIxEhArH](https://openreview.net/forum?id=42dIxEhArH), [NeurIPS virtual poster 155649](https://neurips.cc/virtual/2026/poster/155649) (Paper Copilot `nips2026.json` status = Spotlight)
- 요약 1: 어휘든 시맨틱이든 기존 검색 시스템은 코퍼스 접근을 추론 이전의 단일 top-k 단계로 압축하는 고정된 유사도 인터페이스다. 이 인터페이스는 정확한 어휘 제약, 약한 단서들의 결합, 국소 맥락 확인, 다단계 가설 수정이 필요한 agentic search에서 병목이 되고, 초기에 걸러진 증거는 뒤의 추론으로 되살릴 수 없다. 이를 해결하려고, 에이전트가 임베딩·벡터 인덱스·검색 API 없이 grep, 파일 읽기, 셸 명령, 경량 스크립트 같은 범용 터미널 도구로 원시 코퍼스를 직접 탐색하는 Direct Corpus Interaction(DCI)을 연구한다. DCI는 오프라인 인덱싱이 필요 없고, 변하는 로컬 코퍼스에 자연스럽게 적응한다.
- 요약 2: BRIGHT·BEIR 일부 데이터셋에서 강력한 sparse·dense·reranking 베이스라인을 크게 앞섰고, 기존 시맨틱 리트리버 없이도 BrowseComp-Plus와 멀티홉 QA에서 높은 정확도를 냈다. 검색 품질이 추론 능력뿐 아니라 모델과 코퍼스 사이 인터페이스의 해상도에도 달려 있음을 보였다.
- 인용(OpenReview/NeurIPS판 초록, 미러 기준): "retrieval quality depends not only on reasoning ability, but also on how richly the corpus interface supports interaction". arXiv v1 문구는 "…but also on the resolution of the interface through which the model interacts with the corpus"다.
- 하위분류: 신흥 방향 — 에이전트용 검색 인터페이스 재설계 (리트리버 연구와 경계에 있음)
- 검증: ✅ arXiv abs 확인(제목·1저자·v1·초록 일치; Comments 없음) / ⚠️ venue 출처: Paper Copilot OpenReview 미러만(status=Spotlight, NeurIPS virtual 링크 포함). OpenReview·neurips.cc 원문은 이 세션에서 접속 불가.

#### 같은 방식으로 학회 채택을 확인한 추가 대표 논문 (제목·라벨만; 분류는 Q2에서 사용)
- NeurIPS 2025 — Paper Copilot `nips2025.json` 미러로 재확인(⚠️ OpenReview 원문 미열람). 출처: [papercopilot/paperlists](https://github.com/papercopilot/paperlists)
  - ReSearch: Learning to Reason with Search for LLMs via RL (Mingyang Chen et al.) — Poster
  - WebThinker (Xiaoxi Li et al.) — Poster
  - WebDancer (Jialong Wu et al.) — Poster
  - Chain-of-Retrieval Augmented Generation / CoRAG (Liang Wang et al.) — Poster
  - Improving RAG through Multi-Agent RL / MMOA-RAG (Yiqun Chen et al.) — Poster
  - Search and Refine During Think / AutoRefine (Yaorui Shi et al.) — Poster
  - Process vs. Outcome Reward: Which is Better for Agentic RAG RL (Wenlin Zhang et al.) — Poster
  - Iterative Self-Incentivization Empowers LLMs as Agentic Searchers (Zhengliang Shi et al.) — Poster
  - AceSearcher (Ran Xu et al.) — Spotlight
- EMNLP 2025 Main — ACL Anthology 페이지로 직접 확인 ✅:
  - StepSearch (Xuhui Zheng et al.) — [2025.emnlp-main.1106](https://aclanthology.org/2025.emnlp-main.1106/)
  - s3: You Don't Need That Much Data to Train a Search Agent via RL (Pengcheng Jiang et al.) — [2025.emnlp-main.1095](https://aclanthology.org/2025.emnlp-main.1095/)
  - EvolveSearch: An Iterative Self-Evolving Search Agent (Ding-Chu Zhang et al.) — [2025.emnlp-main.663](https://aclanthology.org/2025.emnlp-main.663/)
  - Search Wisely: Mitigating Sub-optimal Agentic Searches By Reducing Uncertainty (Peilin Wu et al.) — [2025.emnlp-main.998](https://aclanthology.org/2025.emnlp-main.998/)
- ICLR 2026 (Poster):
  - Tree Search for LLM Agent Reinforcement Learning / Tree-GRPO (Yuxiang Ji et al., arXiv [2509.21240](https://arxiv.org/abs/2509.21240)) — arXiv Comments "ICLR 2026" ✅, Paper Copilot `iclr2026.json` status = Poster(OpenReview ZpQwAFhU13), [iclr.cc poster](https://iclr.cc/virtual/2026/poster/10008772)
  - Search Self-Play: Pushing the Frontier of Agent Capability without Supervision (Hongliang Lu et al., arXiv [2510.18821](https://arxiv.org/abs/2510.18821)) — arXiv Comments "Published as a conference paper at … ICLR 2026" ✅, 미러 status = Poster(OpenReview ZmGirmNJqE). 이전 노트의 "Unverified"를 정정했다.
- ICML 2026 (Poster) — Paper Copilot `icml2026.json` 미러로 재확인(⚠️). 출처: [papercopilot/paperlists](https://github.com/papercopilot/paperlists)
  - Stratified GRPO (Mingkang Zhu et al., arXiv [2510.06214](https://arxiv.org/abs/2510.06214), v1 2025-10-07 ✅) — [OpenReview ZX2lOgKOTm](https://openreview.net/forum?id=ZX2lOgKOTm)
  - DECOR: Learning to Decompose and Collaborate in Deep Search via Multi-Agent RL (Ruiqing Chen et al.) — [OpenReview dkRAVqr347](https://openreview.net/forum?id=dkRAVqr347). arXiv 제목 검색(export API)에서는 찾지 못했다.
  - ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards — [OpenReview SeO8pLS44H](https://openreview.net/forum?id=SeO8pLS44H)
  - RE-TRAC: REcursive TRAjectory Compression for Deep Search Agents — [OpenReview wjNTbZKIvC](https://openreview.net/forum?id=wjNTbZKIvC)
  - IntentRL — [OpenReview rWSwu2Req3](https://openreview.net/forum?id=rWSwu2Req3)
  - Optimizing Agentic Reasoning with Retrieval via Synthetic Semantic Information Gain Reward — [OpenReview mUU7k4bN97](https://openreview.net/forum?id=mUU7k4bN97)
- NeurIPS 2026 (Poster) — Paper Copilot `nips2026.json` 미러로 재확인(⚠️). 출처: [papercopilot/paperlists](https://github.com/papercopilot/paperlists)
  - PiCA: Pivot-Based Credit Assignment For Search Agentic RL (Dongyi Liu et al.) — [OpenReview 4CNCefRkpK](https://openreview.net/forum?id=4CNCefRkpK)
  - Beyond Outcome Rewards: Step-Level Self-Distilled PO for Deep Search Agents — [B1ywhnzK7h](https://openreview.net/forum?id=B1ywhnzK7h)
  - Quest: Training Frontier Deep Research Agents with Fully Synthetic Tasks — [eKFW4EIOJx](https://openreview.net/forum?id=eKFW4EIOJx)
  - Deep Research as Rubric — [JmP1iLzzlJ](https://openreview.net/forum?id=JmP1iLzzlJ)
  - RICE-PO: Turning Retrieval Interactions into Credit Signals for Reasoning Agents — [2fIYREq48r](https://openreview.net/forum?id=2fIYREq48r)
  - SD-Search: Hindsight Self-Distillation for Search-Augmented Reasoning — [50CwcISp9W](https://openreview.net/forum?id=50CwcISp9W)
  - SpecHop: Continuous Speculation for Accelerating Multi-Hop Retrieval Agents — [iS6Bu6eqWc](https://openreview.net/forum?id=iS6Bu6eqWc)
  - CARD: Internalizing Expert Critique into RL for Deep Search — [Q15eHqL9AR](https://openreview.net/forum?id=Q15eHqL9AR)
  - Retriever-Free Retrieval-Augmented Reasoning via Corpus-Traversing MCTS — [vi37A4okc6](https://openreview.net/forum?id=vi37A4okc6)
  - CLIFT: Conformal Self-Verification for Web Agent Training and Test-Time Scaling — [igVA4JNEyQ](https://openreview.net/forum?id=igVA4JNEyQ)
- ACL 2026 Main (Long) — ACL Anthology 페이지로 직접 확인 ✅:
  - End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based RL / MHGPO (Guanzhong Chen et al.; arXiv [2506.02718](https://arxiv.org/abs/2506.02718), arXiv Comments "Accepted to ACL 2026 Main Conference"). 현재 arXiv 제목은 Anthology 제목과 같다. 이전 노트의 "v1 제목은 …"은 이번에 확인하지 못했다. — [2026.acl-long.1399](https://aclanthology.org/2026.acl-long.1399/)
  - MemSearch-o1: Empowering LLMs with Reasoning-Aligned Memory Growth in Agentic Search (Sheng Zhang et al.) — [2026.acl-long.41](https://aclanthology.org/2026.acl-long.41/)
  - S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA (Minghan Li et al.) — [2026.acl-long.1185](https://aclanthology.org/2026.acl-long.1185/)
  - Adaptive Retrieval for Reasoning (Jongho Kim et al.; 초록에서 방법명 REPAIR 확인) — [2026.acl-long.1734](https://aclanthology.org/2026.acl-long.1734/)
  - Trust Within? Seek Beyond? Knowledge Boundary Aware PO for Agentic Search (Tao Feng et al.) — [2026.acl-long.1276](https://aclanthology.org/2026.acl-long.1276/)
  - Reinforcing Agentic Search Via Reward Density Optimization (Kun Luo et al.) — [2026.acl-long.467](https://aclanthology.org/2026.acl-long.467/)
  - Chaining the Evidence: Robust RL for Deep Search Agents with Citation-Aware Rubric Rewards (Jiajie Zhang et al.) — [2026.acl-long.950](https://aclanthology.org/2026.acl-long.950/)
  - PRInTS: Reward Modeling for Long-Horizon Information Seeking (Jaewoo Lee et al.) — [2026.acl-long.1574](https://aclanthology.org/2026.acl-long.1574/)
  - D2Plan: Dual-Agent Dynamic Global Planning for Complex Retrieval-Augmented Reasoning (Kangcheng Luo et al.) — [2026.acl-long.216](https://aclanthology.org/2026.acl-long.216/)
- ACL 2026 Findings — ACL Anthology 직접 확인 ✅:
  - BrowseConf: Confidence-Guided Test-Time Scaling for Web Agents — [2026.findings-acl.21](https://aclanthology.org/2026.findings-acl.21/)
  - CriticSearch: Fine-Grained Credit Assignment for Search Agents via a Retrospective Critic — [2026.findings-acl.596](https://aclanthology.org/2026.findings-acl.596/)
  - MemSearcher — [2026.findings-acl.736](https://aclanthology.org/2026.findings-acl.736/)
  - AutoSearch: Adaptive Search Depth for Efficient Agentic RAG via RL — [2026.findings-acl.1399](https://aclanthology.org/2026.findings-acl.1399/)
  - BAPO: Boundary-Aware Policy Optimization for Reliable Agentic Search — [2026.findings-acl.1814](https://aclanthology.org/2026.findings-acl.1814/)
  - Agentic-R: Learning to Retrieve for Agentic Search — [2026.findings-acl.785](https://aclanthology.org/2026.findings-acl.785/)
- EMNLP 2026 Main — ACL Anthology 직접 확인 ✅:
  - Fake It Till You Make It: Training Deep-Search Agents in Synthetic Search Environments (SynSearch) — [2026.emnlp-main.928](https://aclanthology.org/2026.emnlp-main.928/)
  - Dense Process Supervision for Search Agents via Fact Utility Estimation — [2026.emnlp-main.1050](https://aclanthology.org/2026.emnlp-main.1050/)
  - OpenResearcher: A Fully Open Pipeline for Long-Horizon Deep Research Trajectory Synthesis — [2026.emnlp-main.42](https://aclanthology.org/2026.emnlp-main.42/)
  - DeepSearch-World: Self-Distillation for Deep Search Agents in a Verifiable Environment — [2026.emnlp-main.1117](https://aclanthology.org/2026.emnlp-main.1117/)
  - CoPRM: A Collaborative Process Reward Model for Guiding Search Agent Reasoning — [2026.emnlp-main.65](https://aclanthology.org/2026.emnlp-main.65/)
  - To Answer or to Abstain: Mitigating Search-Agent Hallucinations via Abstention-Aware RL — [2026.emnlp-main.161](https://aclanthology.org/2026.emnlp-main.161/)
  - P-Search: A Parallel Plan–Act–Verify Framework with Decoupled GRPO for Deep Research — [2026.emnlp-main.570](https://aclanthology.org/2026.emnlp-main.570/)
  - KnowGPO: Knowledge-Grouped Policy Optimization for Search Agents — [2026.emnlp-main.2357](https://aclanthology.org/2026.emnlp-main.2357/)
  - DEEPRUBRIC: Evidence-Tree Rubric Supervision for Efficient RL of Deep Research Agents (arXiv [2606.17029](https://arxiv.org/abs/2606.17029), Comments "EMNLP 2026 Main Conference") — [2026.emnlp-main.2267](https://aclanthology.org/2026.emnlp-main.2267/)
  - Aggregate While Searching: Online Partial-Trajectory Aggregation for Long-Horizon LLM Search Agents — [2026.emnlp-main.503](https://aclanthology.org/2026.emnlp-main.503/)
- EMNLP 2026 Findings — ACL Anthology 직접 확인 ✅:
  - HALT: Verification-Aware Stopping for Retrieval-Augmented Search Agents — [2026.findings-emnlp.343](https://aclanthology.org/2026.findings-emnlp.343/)
  - One Policy, Any Budget: Internalizing Budget-Aware Search via RL — [2026.findings-emnlp.1857](https://aclanthology.org/2026.findings-emnlp.1857/)
  - Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent — [2026.findings-emnlp.2184](https://aclanthology.org/2026.findings-emnlp.2184/)
  - Beyond Confidence: Test-Time Scaling for Multi-Turn Search Agents via Retrieval Grounding — [2026.findings-emnlp.870](https://aclanthology.org/2026.findings-emnlp.870/)
  - STAMP: Provenance-Guided Credit Assignment for Deep Search Agents — [2026.findings-emnlp.637](https://aclanthology.org/2026.findings-emnlp.637/)
  - APEX-Searcher: Hierarchical Credit Assignment with Subgoaling for Agentic RAG — [2026.findings-emnlp.1538](https://aclanthology.org/2026.findings-emnlp.1538/)
  - Negative Advantage Is a Double-Edged Sword: Calibrating advantages in GRPO for Search Agents — [2026.findings-emnlp.1874](https://aclanthology.org/2026.findings-emnlp.1874/)
  - DeepDiver-DR: Empowering LLMs for Deep Research via Specialized Multi-Agent Collaboration — [2026.findings-emnlp.1154](https://aclanthology.org/2026.findings-emnlp.1154/)
  - Beyond Turn Limits: Training Deep Search Agents with Dynamic Context Window — [2026.findings-emnlp.2461](https://aclanthology.org/2026.findings-emnlp.2461/)

#### 학회 채택이 확인되지 않은 영향력 있는 논문 → "Preprint (arXiv)" 또는 "Unverified"
arXiv abs 페이지로 존재, 제목, 1저자는 모두 확인했다 ✅. 학회 상태는 다음과 같다.
- ZeroSearch: Incentivize the Search Capability of LLMs without Searching (Hao Sun et al., [2505.04588](https://arxiv.org/abs/2505.04588)) — arXiv Comments 없음. ICLR 2026 미러의 status는 **"Desk Reject"**(OpenReview mmkbgNlrJB)다. → **Preprint (arXiv)**
- R1-Searcher (Huatong Song et al., [2503.05592](https://arxiv.org/abs/2503.05592)) — Comments 없음, ICLR 2026 미러에 없음 → **Preprint (arXiv)**
- ASearcher "Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL" (Jiaxuan Gao et al., [2508.07976](https://arxiv.org/abs/2508.07976)) — Comments 없음, ICLR 2026 미러에 없음 → **Preprint (arXiv)**
- Tongyi DeepResearch Technical Report (Tongyi DeepResearch Team, [2510.24701](https://arxiv.org/abs/2510.24701)) — 기술 보고서 → **Preprint (arXiv)**
- CoSearch: Joint Training of Reasoning and Document Ranking via RL for Agentic Search (Hansi Zeng et al., [2604.17555](https://arxiv.org/abs/2604.17555), v1 2026-04-19) → **Preprint (arXiv)**
- To Search or Not to Search: Aligning the Decision Boundary of Deep Search Agents via Causal Intervention (Wenlin Zhang et al., [2602.03304](https://arxiv.org/abs/2602.03304)) → **Preprint (arXiv)**
- Search Self-Play([2510.18821](https://arxiv.org/abs/2510.18821))는 이 목록에서 뺐다. **ICLR 2026 (Poster)**로 정정했다(위 참조).
- SIGIR 2026 논문 2편. arXiv Comments로 채택 문구를 확인했지만, SIGIR 공식 목록은 접속하지 못했다.
  - "When to Retrieve During Reasoning: Adaptive Retrieval for Large Reasoning Models" (Dongxin Guo et al., [2604.26649](https://arxiv.org/abs/2604.26649)) — Comments "Accepted at SIGIR 2026 … Melbourne, Australia" → **SIGIR 2026 (arXiv Comments 기준)**
  - "CroSearch-R1: Better Leveraging Cross-lingual Knowledge for RAG" (Rui Qi et al., [2604.25182](https://arxiv.org/abs/2604.25182)) — Comments "Accepted to SIGIR 2026 (Short Paper)" → **SIGIR 2026 Short (arXiv Comments 기준)**

### Inferences
- 2025년 초 기초 논문 3편(Search-o1, Search-R1, DeepResearcher)은 이후 거의 모든 2026 논문이 비교 기준이나 출발점으로 삼는다. 예를 들어 TreePS-RAG는 초록에서 "Search-R1과 비슷한 롤아웃 비용"을 명시하고, SearchGym은 ASearcher를 비교 기준으로 쓰며, CoSearch는 Search-R1을 "검색 시스템을 고정 도구로 취급하는" 대표 사례로 든다. 따라서 이 셋을 기초로 표시해 포함하는 것이 타당하다.
- 학회 채택 기준으로 보면, 2026년 이 하위 분야의 주 무대는 ACL 계열(ACL 2026, EMNLP 2026)과 ICML/NeurIPS 2026이다. Spotlight급은 deep research 보상 설계(DR Tulu)와 검색 인터페이스 재설계(DCI)처럼 "패러다임 전환"형 논문이다. 단, Spotlight 등급은 OpenReview 미러로만 확인했다.
- 많이 인용되는 산업계 시스템 보고서(Tongyi DeepResearch, ASearcher, ZeroSearch 등)는 확인 가능한 범위에서 학회 채택 기록이 없다. ZeroSearch는 ICLR 2026 미러에서 desk reject로 표시된다. 영향력과 학회 채택이 꼭 일치하지는 않는다.

### Gaps
- OpenReview 원문(결정 문구·등급)은 이번에도 직접 열지 못했다. `openreview.net`과 `api.openreview.net`은 봇 검증 403, `api2.openreview.net`은 프록시 차단, WebFetch는 DNS 실패였다. COLM 2025, ICML 2026, NeurIPS 2025/2026의 결정과 Poster/Spotlight 등급은 Paper Copilot 미러에만 의존한다. 직접 확인하려면 세션 네트워크 설정에서 openreview.net을 허용하고 봇 검증을 통과해야 한다.
- icml.cc, neurips.cc, iclr.cc, colmweb.org, dblp.org, PMLR, NeurIPS proceedings가 모두 프록시 403이어서, 공식 학회 사이트로 교차 확인하지 못했다.
- AAAI 2026, WWW 2026, KDD 2026 공식 채택 목록은 확인하지 못했다. SIGIR 2026은 arXiv Comments로만 확인했다.
- DECOR(ICML 2026)는 arXiv 제목 검색으로도 찾지 못했다. 그래서 큐레이션 목록에는 JADE를 유지했다.
- 보조 목록(추가 대표 논문)의 arXiv v1 일자는 개별 확인하지 않았다. 확인한 것은 Stratified GRPO 2025-10-07과 일부 프리프린트뿐이다.

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
- "Towards Agentic RAG with Deep Reasoning: A Survey of RAG-Reasoning Systems in LLMs"(Yangning Li et al., arXiv [2507.09477](https://arxiv.org/abs/2507.09477) ✅; arXiv Comments "submitted to ARR May", 학회 채택 미확인)의 분류 체계는 공식 저장소에 반영되어 있다. 구성은 다음과 같다.
  - Reasoning-Enhanced RAG(Retrieval Optimization / Integration Enhancement / Generation Enhancement)
  - RAG-Enhanced Reasoning(External Knowledge Retrieval / In-context Retrieval)
  - Synergized RAG and Reasoning: Reasoning Workflow(Chain/Tree/Graph-based)와 Agentic Orchestration(Single-Agent: Prompting·SFT·RL / Multi-Agent)
  - 앞의 두 범주는 "one-way" 강화이고, Synergized 범주는 검색과 추론을 "iteratively" 수행해 서로를 강화한다.
  - 출처: [Awesome-RAG-Reasoning](https://github.com/DavidZWZ/Awesome-RAG-Reasoning)
- "A Comprehensive Survey on Reinforcement Learning-based Agentic Search: Foundations, Roles, Optimizations, Evaluations, and Applications"(Minhua Lin et al., arXiv [2510.16724](https://arxiv.org/abs/2510.16724) ✅)의 저장소는 RL 검색 에이전트를 다음 축으로 분류한다. RL의 기능적 역할, cold start 여부, 학습 환경(Real-world vs Simulated), RL 알고리즘, 보상 유형(ORM vs PRM; 규칙 기반 vs LLM 심판), 최적화 범위(single-agent / multi-agent / module-level). — [Awesome-RL-based-Agentic-Search-Papers](https://github.com/ventr1c/Awesome-RL-based-Agentic-Search-Papers)
- 대표 서베이(arXiv abs에서 제목·1저자 확인 ✅; 목록 출처는 [Awesome-RL-based-Agentic-Search-Papers](https://github.com/ventr1c/Awesome-RL-based-Agentic-Search-Papers)):
  - "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG" (Aditi Singh et al., [2501.09136](https://arxiv.org/abs/2501.09136))
  - "Synergizing RAG and Reasoning: A Systematic Review" (Yunfan Gao et al., [2504.15909](https://arxiv.org/abs/2504.15909))
  - "Deep Research Agents: A Systematic Examination And Roadmap" (Yuxuan Huang et al., [2506.18096](https://arxiv.org/abs/2506.18096))
  - "A Survey of LLM-based Deep Search Agents: Paradigm, Optimization, Evaluation, and Challenges" (Yunjia Xi et al., [2508.05668](https://arxiv.org/abs/2508.05668))
  - "Deep Research: A Systematic Survey" (Zhengliang Shi et al., [2512.02038](https://arxiv.org/abs/2512.02038))
  - "A Survey of Agentic Reasoning for Large Language Models: Towards Recursively Self-Improving and Collective Agents" (Tianxin Wei et al., [2601.12538](https://arxiv.org/abs/2601.12538)) — arXiv Comments "Accepted by TMLR"
  - "The Landscape of Agentic Reinforcement Learning for LLMs: A Survey" (Guibin Zhang et al., [2509.02547](https://arxiv.org/abs/2509.02547)) — arXiv Comments "Published on Transactions on Machine Learning Research"
  - ❌ 삭제: "SoK: Agentic RAG: Taxonomy, Architectures, Evaluation, and Research Directions"([2603.07379](https://arxiv.org/abs/2603.07379))는 arXiv 관리자 공지로 **철회(withdrawn)**되었다: "This submission has been withdrawn because it does not meet arXiv's research content quality standards". 인용 출처에서 제외한다.
- ACL 2026 Main에는 "A Survey of Large Language Model-Based Search Agents"(Yunjia Xi et al., [2026.acl-long.374](https://aclanthology.org/2026.acl-long.374/))가, ACL 2026 Findings에는 "Data-Centric Perspectives on Agentic Retrieval-Augmented Generation: A Survey"(Jingwen Deng et al., [2026.findings-acl.78](https://aclanthology.org/2026.findings-acl.78/))가 채택되었다(Anthology 직접 확인 ✅).

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
  - R1-Searcher (Preprint)
- **B2. RL 알고리즘·보상 설계(과정 보상 / credit assignment / advantage 보정)**
  - [5] TreePS-RAG (EMNLP 2026 Findings)
  - Tree-GRPO (ICLR 2026)
  - Stratified GRPO (ICML 2026)
  - PiCA (NeurIPS 2026)
  - Dense Process Supervision via Fact Utility (EMNLP 2026 Main)
  - Process vs. Outcome Reward (NeurIPS 2025)
  - CriticSearch (ACL 2026 Findings)
  - STAMP, APEX-Searcher (EMNLP 2026 Findings)
- **B3. 학습 환경·데이터(시뮬레이션 / 합성 과제 / self-play)**
  - [6] SearchGym (ACL 2026 Main)
  - SynSearch (EMNLP 2026 Main)
  - DeepSearch-World (EMNLP 2026 Main)
  - OpenResearcher (EMNLP 2026 Main)
  - Quest (NeurIPS 2026)
  - Search Self-Play (ICLR 2026)
  - ZeroSearch (Preprint; ICLR 2026 desk reject)
- **C. Deep-research / 장기 웹 에이전트**
  - [3] DeepResearcher (EMNLP 2025 Main)
  - [4] DR Tulu (ICML 2026 Spotlight)
  - [9] FS-Researcher (ACL 2026 Main)
  - WebThinker, WebDancer (NeurIPS 2025)
  - RE-TRAC (ICML 2026)
  - ASearcher, Tongyi DeepResearch (Preprint)
- **D. 적응형 검색 / 검색 효율(검색 시점·여부·깊이·중단 결정)**
  - [7] Over-Searching (EACL 2026 Main)
  - [8] AdaSearch (EMNLP 2026 Main)
  - Search Wisely (EMNLP 2025 Main)
  - Adaptive Retrieval for Reasoning / REPAIR (ACL 2026 Main)
  - Knowledge Boundary Aware PO (ACL 2026 Main)
  - AutoSearch, BAPO (ACL 2026 Findings)
  - HALT, One Policy Any Budget (EMNLP 2026 Findings)
  - SpecHop (NeurIPS 2026)
  - When to Retrieve During Reasoning (SIGIR 2026, arXiv Comments 기준)
- **E. 멀티에이전트 RAG**
  - [11] JADE (ICML 2026)
  - DECOR (ICML 2026)
  - MHGPO (ACL 2026 Main)
  - MMOA-RAG (NeurIPS 2025)
  - D2Plan (ACL 2026 Main)
  - DeepDiver-DR (EMNLP 2026 Findings)
- **F. Test-time scaling / inference-time compute for RAG**
  - [10] FineVerify (EMNLP 2026 Main)
  - [9] FS-Researcher (ACL 2026 Main)
  - CoRAG (NeurIPS 2025)
  - BrowseConf (ACL 2026 Findings)
  - Beyond Confidence (EMNLP 2026 Findings)
  - CLIFT (NeurIPS 2026)
- **G. (신흥) 에이전트용 검색 인터페이스 재설계**
  - [12] DCI (NeurIPS 2026 Spotlight)
  - Retriever-Free Corpus-Traversing MCTS (NeurIPS 2026)
  - Agentic-R (ACL 2026 Findings)
  - DCI 후속 연구(모두 Preprint; arXiv abs에서 제목·1저자·초록 확인 ✅):
    - GrepSeek: Training Search Agents for Direct Corpus Interaction (Alireza Salemi et al., [2605.29307](https://arxiv.org/abs/2605.29307))
    - Dr-DCI: Scaling Direct Corpus Interaction via Dynamic Workspace Expansion (Yi Lu et al., [2606.14885](https://arxiv.org/abs/2606.14885))
    - A New Role for Relevance: Guiding Corpus Interaction in Agentic Search (Jiangnan Li et al., [2607.24223](https://arxiv.org/abs/2607.24223))
    - Towards Retrieving Interaction Spaces for Agentic Search (Shengyao Zhuang et al., [2606.06880](https://arxiv.org/abs/2606.06880))

### Inferences
- 서베이의 "Prompting / SFT / RL" 축과 "single / multi-agent" 축만으로는 2026년 연구를 충분히 구분하기 어렵다. 2026년 논문의 다수가 RL 학습 에이전트이기 때문이다. 그래서 B 범주를 알고리즘·보상(B2)과 환경·데이터(B3)로 나누는 것이 현재 문헌 분포에 더 맞다.
- C(deep research)와 F(test-time scaling)는 점점 겹치고 있다. FS-Researcher처럼 계산량 배분 자체를 기여로 내세우는 논문이 늘고 있어, 보고서에서는 교차 범주로 표시하는 것이 좋다.
- G는 "리트리버" 연구(다른 조사자 범위)와 경계에 있다. 다만 DCI는 에이전트가 검색 도구를 어떻게 쓰는가를 다루므로 agentic RAG 쪽에 두되, 리트리버 담당자와 중복 확인이 필요하다.
- ACL 2026 Main 서베이 "A Survey of Large Language Model-Based Search Agents"(Yunjia Xi)는 1저자와 주제가 arXiv 2508.05668과 같다. 그 학회판일 가능성이 높지만, 직접 대조하지는 않았다(추정).

### Gaps
- 2026년 서베이 "A Survey of Agentic Reasoning for LLMs"(2601.12538, TMLR)의 원문 분류 체계는 본문을 읽지 않아 반영하지 않았다. 위 분류는 두 awesome 저장소에 반영된 서베이 분류와 학회 채택 논문 분포를 바탕으로 한 제안이다.
- 이전 노트가 의존한 SoK(2603.07379)는 arXiv에서 철회되었으므로, 이를 근거로 한 서술(JADE의 "SoK 인용 사례" 언급)은 삭제했다.
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
- **정량 신호(제목 키워드 매칭; 이전 조사자 집계, 이번에 재집계하지 않음)**: 정규식 `search agent|agentic search|deep research|deep search|agentic rag|agentic retrieval|searcher|search-r1|search-augmented`로 학회 채택 논문 제목을 셌다. 평가·안전 논문도 포함되므로 대략적인 추세 지표다.
  - NeurIPS 2025: 8/6212 → ICML 2026: 19/6341 → NeurIPS 2026: 27/6437 — [papercopilot/paperlists](https://github.com/papercopilot/paperlists)
  - EMNLP 2025 Main: 7/1809 → ACL 2026 Main(long): 31/2222 → EMNLP 2026 Main: 38/2704 — [acl-org/acl-anthology](https://github.com/acl-org/acl-anthology)
  - EMNLP 2025 Findings: 4/1405 → ACL 2026 Findings: 18/2163 → EMNLP 2026 Findings: 35/2478 — [acl-org/acl-anthology](https://github.com/acl-org/acl-anthology)
- **(1) 장문 rubric 보상**
  - DR Tulu(ICML 2026 Spotlight)는 RLVR이 장문 과제로 확장되지 않는 문제를, 정책과 공진화하는 루브릭으로 해결한다. — [arXiv 2511.19399](https://arxiv.org/abs/2511.19399)
  - 후속 채택작:
    - "Chaining the Evidence: … Citation-Aware Rubric Rewards"(ACL 2026 Main, [2026.acl-long.950](https://aclanthology.org/2026.acl-long.950/))
    - DEEPRUBRIC(EMNLP 2026 Main, [2026.emnlp-main.2267](https://aclanthology.org/2026.emnlp-main.2267/)). LLM이 쿼리에서 루브릭을 추론하는 기존 방식이 불완전한 루브릭을 만든다는 점을 지적하고, "증거 기반 보고서가 무엇으로 평가되어야 하는지"를 먼저 정한 뒤 정렬된 쿼리–루브릭 쌍을 합성한다. — [arXiv 2606.17029](https://arxiv.org/abs/2606.17029)
    - "Deep Research as Rubric"(NeurIPS 2026, [JmP1iLzzlJ](https://openreview.net/forum?id=JmP1iLzzlJ))
    - LongTraceRL(EMNLP 2026 Findings, [2026.findings-emnlp.1222](https://aclanthology.org/2026.findings-emnlp.1222/))
  - HOTE(Hybrid Open-Ended Tri-Evolution, [arXiv 2606.13710](https://arxiv.org/abs/2606.13710), Preprint)는 proposer·solver·judge를 하이브리드 RL로 공진화시켜 개방형 deep research 과제에 적용한다. 이전 노트의 "DR Tulu를 기반으로 확장"이라는 서술은 HOTE·DEEPRUBRIC 초록에서 확인되지 않아 삭제했다.
- **(2) 과정 보상 / credit assignment**
  - TreePS-RAG: 롤아웃 트리와 몬테카를로 단계 효용 — [2026.findings-emnlp.1007](https://aclanthology.org/2026.findings-emnlp.1007/)
  - Tree-GRPO(ICLR 2026): 트리 탐색 기반 그룹 RL로, outcome 보상만으로도 단계별 과정 신호를 얻는다. — [arXiv 2509.21240](https://arxiv.org/abs/2509.21240), [iclr.cc](https://iclr.cc/virtual/2026/poster/10008772)
  - PiCA(NeurIPS 2026): 정보량이 크게 늘어나는 "pivot" 단계를 찾고 PBRS(Potential-Based Reward Shaping) 기반으로 보상한다. 7개 지식집약 QA 벤치마크에서 3B/7B 모델이 각각 15.2%/2.2% 향상했다(OpenReview 초록, 미러 기준). — [OpenReview](https://openreview.net/forum?id=4CNCefRkpK)
  - Stratified GRPO(ICML 2026): 도구 호출 구조별로 계층화한 advantage 정규화. GRPO 대비 향상 폭은 판본마다 다르다. OpenReview(ICML)판 초록은 "최대 12.6점"(factual QA와 deep-research 벤치마크), arXiv 현재판 초록은 "최대 11.3점"(단일·멀티홉 QA)이다. — [OpenReview](https://openreview.net/forum?id=ZX2lOgKOTm), [arXiv 2510.06214](https://arxiv.org/abs/2510.06214)
  - Dense Process Supervision via Fact Utility(EMNLP 2026 Main) — [2026.emnlp-main.1050](https://aclanthology.org/2026.emnlp-main.1050/)
  - STAMP: Provenance-Guided Credit Assignment(EMNLP 2026 Findings) — [2026.findings-emnlp.637](https://aclanthology.org/2026.findings-emnlp.637/)
  - 2025년 NeurIPS의 "Process vs. Outcome Reward" 논문이 이 흐름의 출발점에 있다. — [papercopilot/paperlists](https://github.com/papercopilot/paperlists)
- **(3) 과잉 검색 억제 / 적응형·예산 인지 검색**
  - Over-Searching(EACL 2026 Main): 과잉 검색이 복잡 추론 모델과 deep research 시스템에서 더 심하다는 실증과 TPC 지표 — [2026.eacl-long.361](https://aclanthology.org/2026.eacl-long.361/)
  - AdaSearch(EMNLP 2026 Main): 검색 결정을 문제 해결과 분리 — [2026.emnlp-main.2696](https://aclanthology.org/2026.emnlp-main.2696/)
  - 관련 채택작:
    - AutoSearch "Adaptive Search Depth"(ACL 2026 Findings, [2026.findings-acl.1399](https://aclanthology.org/2026.findings-acl.1399/))
    - HALT "Verification-Aware Stopping"([2026.findings-emnlp.343](https://aclanthology.org/2026.findings-emnlp.343/))
    - "One Policy, Any Budget: Internalizing Budget-Aware Search via RL"([2026.findings-emnlp.1857](https://aclanthology.org/2026.findings-emnlp.1857/))
    - "Not Worth Another Token: Marginal Value Estimation for Efficient Deep Research Agents"([2026.findings-emnlp.2218](https://aclanthology.org/2026.findings-emnlp.2218/))
    - "To Answer or to Abstain"(EMNLP 2026 Main, [2026.emnlp-main.161](https://aclanthology.org/2026.emnlp-main.161/))
- **(4) 시뮬레이션·합성 검색 환경**
  - SearchGym(ACL 2026 Main)은 지식그래프와 정렬 코퍼스로 "strictly solvable" 과제를 만들고, 웹 기반 ASearcher보다 평균 상대 10.6% 높다. — [2026.acl-long.848](https://aclanthology.org/2026.acl-long.848/)
  - SynSearch(EMNLP 2026 Main)는 환경과 과제 분포를 공동 설계한 완전 합성 search–browse 환경으로, 실제 검색 평가에서도 강한 성능을 보인다. 이 설명은 이전 조사자의 Anthology 초록 요약이며, 이번에는 제목·1저자·볼륨만 재확인했다. — [2026.emnlp-main.928](https://aclanthology.org/2026.emnlp-main.928/)
  - 관련 채택작: DeepSearch-World(EMNLP 2026 Main, [2026.emnlp-main.1117](https://aclanthology.org/2026.emnlp-main.1117/)), Quest(NeurIPS 2026, [eKFW4EIOJx](https://openreview.net/forum?id=eKFW4EIOJx)), OffSeeker "Online RL Is Not All You Need for Deep Research Agents"(EMNLP 2026 Main, [2026.emnlp-main.173](https://aclanthology.org/2026.emnlp-main.173/)), Search Self-Play(ICLR 2026, [arXiv 2510.18821](https://arxiv.org/abs/2510.18821))
- **(5) 장기 컨텍스트·메모리 관리**
  - FS-Researcher(ACL 2026 Main): 파일 시스템을 외부 메모리로 사용 — [2026.acl-long.288](https://aclanthology.org/2026.acl-long.288/)
  - MemSearch-o1(ACL 2026 Main): think–search 루프에서 생기는 "memory dilution"을 다룬다(이전 조사자 요약). — [2026.acl-long.41](https://aclanthology.org/2026.acl-long.41/)
  - RE-TRAC(ICML 2026, [wjNTbZKIvC](https://openreview.net/forum?id=wjNTbZKIvC)), ARC: Active and Reflection-driven Context Management(ACL 2026 Findings, [2026.findings-acl.930](https://aclanthology.org/2026.findings-acl.930/)), "Beyond Turn Limits: … Dynamic Context Window"([2026.findings-emnlp.2461](https://aclanthology.org/2026.findings-emnlp.2461/))
- **(6) agentic search의 test-time scaling / 자기검증**
  - FineVerify(EMNLP 2026 Main): 샘플 12개로 GPT-5-mini가 BrowseComp-Plus에서 GPT-5를 넘어섰다. — [2026.emnlp-main.2599](https://aclanthology.org/2026.emnlp-main.2599/)
  - BrowseConf(ACL 2026 Findings, [2026.findings-acl.21](https://aclanthology.org/2026.findings-acl.21/)), "Inference-Time Scaling of Verification: Self-Evolving Deep Research Agents via Test-Time Rubric-Guided Verification"(ACL 2026 Findings, [2026.findings-acl.1243](https://aclanthology.org/2026.findings-acl.1243/)), CLIFT(NeurIPS 2026, [igVA4JNEyQ](https://openreview.net/forum?id=igVA4JNEyQ))
- **(7) 검색 인터페이스·리트리버와 에이전트의 공동 설계**
  - DCI(NeurIPS 2026 Spotlight): grep/셸로 코퍼스를 직접 탐색하는 방식이 sparse/dense/rerank 베이스라인을 앞선다. — [arXiv 2605.05242](https://arxiv.org/abs/2605.05242)
  - "Towards Retrieving Interaction Spaces"(Preprint, v1 2026-06-05)는 무제한 상호작용이 확장되지 않는다고 지적한다. 광범위한 셸 명령은 매번 전체 코퍼스를 스캔하므로 코퍼스가 커질수록 지연이 급격히 나빠진다. 그래서 검색의 역할을 "에이전트가 도구로 탐색할 수 있는 코퍼스의 경계 있는 부분집합(interaction space)을 구성하는 것"으로 재정의한다(arXiv 초록으로 확인 ✅). — [arXiv 2606.06880](https://arxiv.org/abs/2606.06880)
  - Dr-DCI(Preprint)는 검색을 로컬 작업공간 확장 행동으로 쓰고, RARG("A New Role for Relevance", Preprint)는 관련도를 grep 실행 우선순위로 쓴다. 둘 다 DCI의 확장성 한계를 보완한다(arXiv 초록 확인 ✅). — [arXiv 2606.14885](https://arxiv.org/abs/2606.14885), [arXiv 2607.24223](https://arxiv.org/abs/2607.24223)
  - CoSearch(Preprint)는 추론 에이전트와 생성형 문서 랭커를 GRPO로 공동 학습한다. Search-R1류가 검색 시스템을 고정 도구로 취급한다는 점을 문제로 들며, 예비 실험에서 오라클 검색과 고정 검색의 격차가 7개 QA 벤치마크에서 최대 +26.8% 상대 F1에 이른다고 보고한다(arXiv 초록 확인 ✅). — [arXiv 2604.17555](https://arxiv.org/abs/2604.17555)
- **(8) 멀티에이전트 공동 RL** (보조 방향)
  - JADE(ICML 2026)
  - DECOR(ICML 2026): deep search를 MARL로 정식화하고 역할 특화 에이전트를 학습한다. 미러 초록 기준이며, 단일 에이전트의 "cognitive overload"와 동결 모델 기반 멀티에이전트의 한계를 지적한다. — [OpenReview dkRAVqr347](https://openreview.net/forum?id=dkRAVqr347)
  - MHGPO(ACL 2026 Main; critic 없는 이종 그룹 상대 advantage) — [2026.acl-long.1399](https://aclanthology.org/2026.acl-long.1399/)

### Inferences
- 2025년의 핵심 질문은 "RL로 검색을 학습시킬 수 있는가"(Search-R1)였다. 2026년에는 "무엇을 보상하고(rubric·과정 보상), 어디서 학습하고(시뮬레이션), 얼마나 검색할 것인가(과잉 검색·예산)"로 옮겨 갔다. outcome-only 보상의 한계와 실제 웹 API의 비용·비재현성이 공통 동기다.
- 학회 채택 수로 보면 2026년 ACL·EMNLP의 Main과 Findings 모두에서 이 하위 분야가 크게 늘었다. 키워드 카운트는 대략적이지만 증가 폭이 몇 배에 이르므로 추세 판단에는 충분하다고 본다.
- "검색 인터페이스 재설계"(DCI)와 "리트리버-에이전트 공동 학습"(CoSearch, Agentic-R)은 2026년 후반의 새 흐름으로 보인다. DCI 후속 프리프린트 4편이 2026년 5–7월에 연달아 나왔지만 모두 Preprint이고 학회 채택은 DCI와 Agentic-R 정도다. 정착된 방향이라기보다 초기 신호로 다루는 것이 안전하다.

### Gaps
- ICLR 2027 제출작(심사 중, 결정 없음)과 2026년 9–11월 arXiv 신작은 학회 검증이 불가능하므로 체계적으로 조사하지 않았다. 최신 프리프린트 예시(arXiv 초록 확인 ✅, 모두 Preprint (arXiv)):
  - "Contextual Information Policy Optimization for Search Agents"(CIPO; Xingyu Guo et al., [2608.06128](https://arxiv.org/abs/2608.06128), v1 2026-08-06): 검색 이후 행동이 검색 증거에 근거하는지를 직접 보상해 "확증 편향적, 사전지식 주도 추론"을 줄인다.
  - "SOAR: Source-Aware RL with Precise Credit Assignment for Search Agents"(Shengze Huang et al., [2610.13157](https://arxiv.org/abs/2610.13157), v1 2026-10-09): 외부/내부 지식 흐름을 분리한 출처 명시 궤적과 Source-Consistent Reward로 "증거 날조(evidence fabrication)"를 줄인다.
- 키워드 카운트에는 평가·안전·멀티모달 논문(범위 밖)이 섞여 있다. 하위 범주별 정밀 집계는 하지 않았고, 이번 검증에서 카운트를 재집계하지도 않았다.
- AAAI 2026, WWW 2026, KDD 2026의 채택 논문은 공식 목록에 접근하지 못해 추세 집계에서 빠졌다.

## Verification log
검증일: 2026-11-19. 직접 확인 경로는 arxiv.org/abs와 aclanthology.org다. OpenReview는 Paper Copilot 미러로 대체했다.

**존재 확인 결과**
- 큐레이션 12편 모두 arXiv에 존재하며, 제목과 1저자가 노트와 일치한다. 삭제한 논문은 없다.
- 본문의 기타 arXiv ID 30개와 ACL Anthology ID 53개도 모두 존재하며, 제목이 노트와 일치한다.

**정정 사항**
1. **최초 공개일**: 12편 모두 YYMM에서 정확한 v1 일자로 바꿨다.
   - 2025-01-09, 2025-03-12, 2025-04-04, 2025-11-24
   - 2026-01-11, 2026-01-21, 2026-01-09, 2025-12-18
   - 2026-02-02, 2026-05-30, 2026-01-29, 2026-05-03
   - FineVerify는 ID가 2606이지만 v1은 2026-05-30이다.
2. **venue 1차 근거 추가 (arXiv Comments)**:
   - DR Tulu "ICML 2026"
   - Over-Searching "EACL 2026 Main"
   - AdaSearch "EMNLP 2026"
   - FS-Researcher "ACL 2026"
   - FineVerify "EMNLP 2026 main"
   - MHGPO "ACL 2026 Main"
   - DEEPRUBRIC "EMNLP 2026 Main"
   - Tree-GRPO "ICLR 2026"
3. **Search Self-Play (2510.18821)**: "Unverified"를 **ICLR 2026 (Poster)**로 정정했다. 근거는 arXiv Comments와 ICLR 2026 미러다.
4. **SIGIR 2026 두 편 (2604.26649, 2604.25182)**: "Unverified"를 **SIGIR 2026 (arXiv Comments 기준)**으로 정정했다. CroSearch-R1은 Short Paper다.
5. **ZeroSearch**: ICLR 2026 미러에서 "Desk Reject"로 확인했다. **Preprint (arXiv)**로 명시했다.
6. **SoK: Agentic RAG (2603.07379)**: arXiv 관리자가 철회한 논문이다. 서베이 목록과 JADE 항목의 인용에서 ❌ 제거했다.
7. **서베이 제목 정정**:
   - 2601.12538의 실제 제목은 "A Survey of Agentic Reasoning for Large Language Models: Towards Recursively Self-Improving and Collective Agents"이며 TMLR 채택이다.
   - 2501.09136의 정확한 제목은 "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG"다.
   - 2507.09477의 Comments는 "submitted to ARR May"이고, 학회 채택은 미확인이다.
8. **인용구 수정**:
   - JADE: 노트의 인용구는 원문을 줄인 문구였다. arXiv 원문 "within the capability boundaries of the executors"로 바꿨다.
   - DeepResearcher·Search-o1·DCI: 노트의 인용구가 학회판(Anthology 또는 OpenReview) 초록 문구와 같음을 확인했다. arXiv판 문구를 병기했다.
9. **요약 보완 (초록 반영)**:
   - Over-Searching: 발견 (iii)(부정 증거가 기권을 개선)과 완화책 검토를 추가했다.
   - AdaSearch: 기존 tool-call 보상 셰이핑의 한계와 F1 결정 지표의 발견을 추가했다.
   - JADE: 복잡도가 늘어도 성능이 떨어질 수 있다는 점을 추가했다.
   - DCI: 오프라인 인덱싱이 필요 없다는 점을 추가했다.
   - FineVerify: 검증 기록을 감사에 활용할 수 있다는 점을 추가했다.
   - SearchGym: 커리큘럼 단계를 추가했다.
   - Search-R1: 41%/20% 수치의 출처를 "COLM 판"에서 "arXiv 최신판 초록과 동일"로 바꿨다.
10. **Stratified GRPO 수치**: 노트의 "최대 12.6점"은 OpenReview(ICML)판 수치다. arXiv 현재판은 "최대 11.3점"이므로 둘 다 병기했다.
11. **Q3 rubric 후속 연구**: "DR-Rubric/DeepRubric/HOTE가 DR Tulu 기반 확장"이라는 서술을 삭제했다. 초록에서 확인되지 않았다. 대신 각 초록 내용으로 대체했다.
12. **"Towards Retrieving Interaction Spaces"와 CoSearch**: 이전에 웹 검색 스니펫에 의존했던 서술을 arXiv 초록으로 재확인했다(✅). CoSearch의 +26.8% 수치를 추가했다.
13. **보조 목록의 축약 제목**: Anthology 메타데이터로 1저자와 정식 제목을 보완했다(EvolveSearch, Search Wisely, D2Plan, REPAIR 등). JADE와 동명인 ICML 2026 논문(SoilRyCv1i)에 대한 주의도 추가했다.

**여전히 ⚠️/Unverified인 항목**
- OpenReview 기반 결정은 모두 Paper Copilot 미러에만 의존한다. OpenReview 원문은 봇 검증 403과 DNS 차단으로 열지 못했다. 해당 항목:
  - Search-R1: COLM 2025 Poster
  - DR Tulu: Spotlight 등급
  - JADE: ICML 2026 Poster
  - DCI: NeurIPS 2026 Spotlight
  - NeurIPS 2025, ICML 2026, NeurIPS 2026 보조 목록 전체
- MHGPO의 "v1 제목" 주장은 재확인하지 못했다.
- DECOR는 arXiv에서 찾지 못했다.
- Q3 키워드 카운트는 재집계하지 않았다.
