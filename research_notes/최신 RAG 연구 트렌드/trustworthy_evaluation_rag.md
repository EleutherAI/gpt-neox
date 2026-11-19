# 신뢰할 수 있는 RAG(Trustworthy RAG) 및 RAG 평가/벤치마크 — 최신 논문 동향 (기준일: 2026-11-19)

> **검증 방법에 관한 중요 고지 (리포트 작성자 필독)**
> - 이 세션에서는 `arxiv.org`, `aclanthology.org`, `mlanthology.org` 직접 접속(WebFetch 및 arXiv API)이 네트워크 정책으로 차단되었다(`getaddrinfo ENOTFOUND` / proxy `connect_rejected`). 따라서 모든 서지 정보·게재처·초록 내용은 **검색 엔진이 색인한 1차 페이지(arXiv abs/HTML 페이지, USENIX 프로그램 페이지, ACM DL DOI, ACL Anthology URL, iclr.cc 포스터 페이지)의 스니펫**을 통해 확인했다. 초록 전문을 직접 열람하지는 못했다.
> - 따라서 아래 "인용구(quote)"는 검색 스니펫에 나타난 초록 문구이며, 최종 보고서 게재 전에 arXiv 원문과 한 번 더 대조하는 것을 권장한다. 각 항목에 검증 수준을 표기했다.
> - 게재처(venue) 라벨은 1차 출처(arXiv Comments 필드, 학회 공식 프로그램 페이지, ACL Anthology 볼륨 ID, ACM DOI, iclr.cc 포스터 페이지)로 확인된 경우에만 붙였다. 그 외는 "Preprint (arXiv)"로 표기했다.

## Q1. 2025–2026년 RAG 신뢰성·평가 분야의 대표 논문은 무엇이며, 각각 어느 학회에 채택되었나?

### Takeaway
2025–2026년의 대표 흐름은 (1) PoisonedRAG(USENIX Security 2025)를 기점으로 한 코퍼스 오염·간접 프롬프트 인젝션 공격의 정교화(CCS 2026, USENIX Security 2026, WWW 2026)와 그에 맞선 "격리(isolation)" 기반 방어, (2) 인용·귀속(attribution)과 충실성(faithfulness) 검증의 정식화(ACL 2026 Main/Findings), (3) RAG 평가가 "짧은 QA 정답률"에서 "nugget 기반 자동평가(SIGIR 2025)"와 "딥리서치 보고서 평가(ICLR 2026, 2026 프리프린트)"로 이동한 것이다. 아래 14편(기초 논문 1편 포함)을 선정했다.

### Cited Findings

#### 선정 논문 목록 (서브카테고리 순)

**[A] 강건성 & 지식 충돌 (Robustness / Knowledge Conflict)**

**A1. Retrieval-Augmented Generation with Conflicting Evidence** (MADAM-RAG / RAMDocs)
- 저자: Han Wang et al. (Han Wang, Archiki Prasad, Elias Stengel-Eskin, Mohit Bansal; UNC Chapel Hill) — [arXiv 2504.13079](https://arxiv.org/abs/2504.13079v2)
- arXiv: 2504.13079 — https://arxiv.org/abs/2504.13079 · 최초 공개: 2025-04 (ID 기준; 정확한 일자 미확인)
- 게재처: **COLM 2025** (arXiv 목록에 COLM 2025로 표기) — [arXiv 2504.13079v2](https://arxiv.org/abs/2504.13079v2)
- 요약(한국어):
  - 검색 문서에 모호성·오정보·노이즈가 동시에 섞인 현실적 상황을 다루기 위해 RAMDocs 데이터셋을 만들고, 문서별 에이전트가 여러 라운드 토론한 뒤 집계자가 답을 모으는 다중 에이전트 기법 MADAM-RAG를 제안한다.
  - AmbigDocs·RAMDocs에서 Astute RAG 등 기존 기법보다 성능이 높지만(AmbigDocs에서 최대 11.40% 향상 보고), 오정보가 지지 증거보다 많아질 때는 여전히 큰 격차가 남는다고 보고한다. — [arXiv PDF](https://arxiv.org/pdf/2504.13079); [arXiv HTML v1](https://arxiv.org/html/2504.13079v1)
- 인용구: "RAMDocs (Retrieval with Ambiguity and Misinformation in Documents)" (검증 수준: 검색 스니펫 + 기억 기반, 원문 대조 권장)

**[B] 충실성·환각 탐지·인용/귀속 (Faithfulness / Hallucination / Attribution)**

**B1. Faithfulness-Aware Uncertainty Quantification for Fact-Checking the Output of Retrieval Augmented Generation** (FRANQ)
- 저자: Ekaterina Fadeeva et al. (공저자 9명 추가) — [arXiv 2505.21072](https://arxiv.org/abs/2505.21072v2)
- arXiv: 2505.21072 — https://arxiv.org/abs/2505.21072 · 최초 공개: **2025-05-27** (v3: 2026-01-08)
- 게재처: **ACL 2026 Findings** (ACL Anthology ID `2026.findings-acl.338`) — [ACL Anthology](https://aclanthology.org/2026.findings-acl.338/)
- 요약(한국어):
  - RAG 출력의 각 주장(claim)이 검색 문맥에 충실한지(faithfulness)를 먼저 판정하고, 그 결과에 따라 서로 다른 불확실성 정량화(UQ) 기법을 적용해 사실 오류(환각)를 탐지하는 방법을 제안한다. 문맥이 뒷받침하지 않아도 사실일 수 있으므로 "사실성"과 "충실성"을 분리한다는 것이 핵심 아이디어다.
  - 사실성·충실성 이중 라벨이 달린 장문 QA 데이터셋을 새로 구축했고, 기존 방법보다 RAG 응답의 사실 오류를 더 정확히 탐지한다고 보고한다. — [arXiv 2505.21072](https://arxiv.org/abs/2505.21072v2)
- 인용구: "achieves more accurate detection of factual errors in RAG-generated responses compared to existing approaches" (검증 수준: 검색 스니펫)

**B2. CiteGuard: Faithful Citation Attribution for LLMs via Retrieval-Augmented Validation**
- 저자: Yee Man Choi et al. (Yee Man Choi, Xuehang Guo, Yi R. (May) Fung, Qingyun Wang) — [arXiv 2510.17853](https://arxiv.org/pdf/2510.17853)
- arXiv: 2510.17853 — https://arxiv.org/abs/2510.17853 · 최초 공개: 2025-10 (ID 기준)
- 게재처: **ACL 2026 Main (Long)** (ACL Anthology ID `2026.acl-long.282`, pp. 6241–6257) — [ACL Anthology](https://aclanthology.org/2026.acl-long.282/)
- 요약(한국어):
  - 기존 LLM-as-a-Judge 기반 인용 평가는 생성에 쓰인 검색 지식이 주어져 있다고 가정한다는 한계를 지적하고, 인용 평가를 "인용 귀속 정렬(citation attribution alignment)" 문제로 재정의한 뒤 문헌 검색 에이전트(Semantic Scholar API 기반)로 인용-주장 적합성을 검증하는 CiteGuard를 제안한다.
  - ACL 2026 버전 기준 CiteME 벤치마크에서 기존 기준선 대비 10%p 향상, 최대 68.1% 정확도로 인간 성능(69.2%)에 근접했으며, 대체 가능한 유효 인용도 찾아낸다고 보고한다. (arXiv v1은 65.4% vs 인간 69.7%로 수치가 다름 — 게재본 수치 사용 권장) — [ACL Anthology](https://aclanthology.org/2026.acl-long.282/); [arXiv HTML v3](https://arxiv.org/html/2510.17853v3)
- 인용구: "improves over the prior baseline by 10 percentage points and achieves up to 68.1% accuracy on the CiteME benchmark" (검증 수준: 검색 스니펫, ACL 게재본)

**[C] 보안: 공격 (Security — Corpus Poisoning / Indirect Prompt Injection)**

**C1. [기초 논문 / Foundational] PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models**
- 저자: Wei Zou et al. (Wei Zou, Runpeng Geng, Binghui Wang, Jinyuan Jia) — [USENIX](https://www.usenix.org/conference/usenixsecurity25/presentation/zou)
- arXiv: 2402.07867 — https://arxiv.org/abs/2402.07867 · 최초 공개: 2024-02 (ID 기준)
- 게재처: **USENIX Security 2025** (34th USENIX Security Symposium, pp. 3827–3844) — [USENIX 발표 페이지](https://www.usenix.org/conference/usenixsecurity25/presentation/zou); [공식 PDF](https://usenix.org/system/files/usenixsecurity25-zou-poisonedrag.pdf)
- 요약(한국어):
  - RAG의 지식 DB에 소수의 악성 텍스트를 주입해 공격자가 고른 질문에 대해 LLM이 공격자가 고른 답을 내도록 만드는 "지식 오염(knowledge corruption)" 공격을 최적화 문제로 정식화한다.
  - 수백만 건 규모의 지식 DB에 질문당 악성 텍스트 5개만 주입해도 90% 공격 성공률을 달성했고, 평가한 여러 방어가 불충분함을 보였다 — 이후 RAG 보안 연구의 기준 위협 모델이 되었다. — [arXiv 2402.07867](https://arxiv.org/pdf/2402.07867)
- 인용구: "a 90% attack success rate when injecting five malicious texts for each target question" (검증 수준: 검색 스니펫 + 기억, 수치는 스니펫과 일치)

**C2. Overcoming the Retrieval Barrier: Indirect Prompt Injection in the Wild for LLM Systems**
- 저자: Hongyan Chang et al. (Hongyan Chang, Ergute Bao, Xinjian Luo, Ting Yu; MBZUAI) — [arXiv 2601.07072](https://arxiv.org/abs/2601.07072v1)
- arXiv: 2601.07072 — https://arxiv.org/abs/2601.07072 · 최초 공개: **2026-01-11**
- 게재처: **USENIX Security 2026** — [USENIX Security '26 발표 페이지](https://www.usenix.org/conference/usenixsecurity26/presentation/chang); [prepub PDF](https://www.usenix.org/system/files/conference/usenixsecurity26/sec26_prepub_chang.pdf)
- 요약(한국어):
  - 간접 프롬프트 인젝션이 실제로는 "검색되어야만" 작동한다는 장벽에 주목해, 악성 문서를 검색을 보장하는 트리거 조각과 임의의 공격 목표를 담는 공격 조각으로 분해하는 방법을 제안한다(임베딩 모델 API 접근만 필요).
  - 11개 벤치마크·8개 임베딩 모델에서 거의 100% 검색 성공, 쿼리당 최저 $0.21 비용을 보고하며, 다중 에이전트 워크플로에서 오염된 이메일 1통으로 GPT-4o가 SSH 키를 유출하게 만드는 데 80% 이상 성공했다; 평가한 방어들은 악성 텍스트 검색을 막기에 불충분했다. — [arXiv 2601.07072](https://arxiv.org/pdf/2601.07072); [USENIX](https://www.usenix.org/conference/usenixsecurity26/presentation/chang-hongyan)
- 인용구: "achieves near-100% retrieval across 11 benchmarks and 8 embedding models" (검증 수준: 검색 스니펫, USENIX/arXiv 양쪽 일치)

**C3. Hidden-in-Plain-Text: A Benchmark for Social-Web Indirect Prompt Injection in RAG** (OpenRAG-Soc)
- 저자: Haoze Guo et al. (Haoze Guo, Ziqi Wei; UW–Madison) — [arXiv 2601.10923](https://arxiv.org/abs/2601.10923)
- arXiv: 2601.10923 — https://arxiv.org/abs/2601.10923 · 최초 공개: **2026-01-16**
- 게재처: **WWW 2026 (The ACM Web Conference 2026)** — [ACM DL DOI 10.1145/3774904.3792853](https://dl.acm.org/doi/10.1145/3774904.3792853) (트랙 구분 — main/resource/short — 은 미확인)
- 요약(한국어):
  - 사용자 생성 웹 콘텐츠에 숨겨진 지시문(hidden span, 화면 밖 CSS, alt 텍스트, ARIA 속성, zero-width 문자)을 통한 간접 프롬프트 인젝션과 검색 오염을 RAG 전체 파이프라인(수집→검색→생성) 관점에서 측정하는 벤치마크 OpenRAG-Soc를 제안한다.
  - 6,200개의 소셜 스타일 웹 페이지 코퍼스와 정상 명령문 hard negative를 포함하며, 숨김 HTML/Markdown 정화, 유니코드 정규화, 귀속 기반 응답(attribution-gated answering) 등 실용적 완화책을 함께 평가한다. — [arXiv 2601.10923](https://arxiv.org/pdf/2601.10923)
- 인용구: "hidden spans, off-screen CSS, alt text, ARIA attributes, and zero-width characters" (검증 수준: 검색 스니펫; 한 2차 사이트(deeppaper v2)가 다른 논문 내용을 섞어 표기하므로 원문 대조 필수)

**C4. InceptionRAG: Stealthy Poisoning Attack Against Retrieval-Augmented Generation**
- 저자: Jiachang Zhang et al. (Jiachang Zhang, Min Chen, Xiao Ren, Zhenyong Zhang, Yuanchao Shu, Yunjun Gao, Zhikun Zhang) — [arXiv 2609.16818](https://arxiv.org/abs/2609.16818v1)
- arXiv: 2609.16818 — https://arxiv.org/abs/2609.16818 · 최초 공개: **2026-09-15**
- 게재처: **ACM CCS 2026** (arXiv Comments에 CCS '26 채택 표기) — [arXiv 2609.16818v1](https://arxiv.org/abs/2609.16818v1)
- 요약(한국어):
  - 각각은 무해해 보이는 두 개의 "잠복" 문서의 논리적 근접성을 이용해, 거짓 결론이 코퍼스에 명시적으로 쓰이지 않고 생성 단계에서만 합성되도록 하는 은밀한 오염 공격을 제안한다 — 명시적 독성 콘텐츠를 찾는 기존 방어를 우회한다.
  - 추론 기반 오염으로 80% 이상의 공격 성공률을 보고하고, 적대적 논리 의존성을 끊는 문서 격리 방어 HODOR(문서별 개별 응답 후 다수결)를 제안했으나 이는 다중 홉 추론 과제의 정상 정확도를 떨어뜨린다. — [arXiv HTML](https://arxiv.org/html/2609.16818v1); 80% 수치는 [검색 요약](https://arxiv.org/abs/2609.16818v1) 기준
- 인용구: "a document isolation defense that decouples the adversarial logical dependencies" (검증 수준: 검색 스니펫)

**[D] 보안: 방어 및 방어 평가의 재검토 (Defenses & Defense Meta-evaluation)**

**D1. Cordon-MAS: Defending RAG against Knowledge Poisoning via Information-Flow Control**
- 저자: Zhe Yu et al. (Zhe Yu, Wenpeng Xing, Gaolei Li, Shuguang Xiong, Hongzhi Wang, Xuyang Teng, Meng Han) — [arXiv 2605.26754](https://arxiv.org/abs/2605.26754) (다른 논문들이 다른 저자 목록으로 인용한다는 스니펫이 있어 arXiv 원문 확인 권장)
- arXiv: 2605.26754 — https://arxiv.org/abs/2605.26754 · 최초 공개: 2026-05 (ID 기준)
- 게재처: **Preprint (arXiv)** (Comments에서 채택 정보 확인되지 않음)
- 요약(한국어):
  - 모델이 검색 증거의 모순을 "탐지"하고도 오염된 주장에 따라 "행동"하는 monitoring-control gap을 지적하고, 탐지 대신 정보흐름 제어(IFC)를 적용한 다중 에이전트 구조(Extractor가 구조화된 claim 카드 생성 → Auditor 교차검증 → 원문을 전혀 보지 않는 Synthesizer가 답변)를 제안한다.
  - 5개 BEIR 데이터셋에서 방어 없는 RAG 대비 공격 성공률을 92.4% 감소시켰다고 보고하며(학습형 오염 공격 Confundo 등 적응형 공격 대상), 대신 질의당 여러 번의 순차 LLM 호출이 필요하다. — [arXiv 2605.26754 HTML](https://arxiv.org/html/2605.26754v1); 비용 지적은 [TriShieldRAG 2607.23838](https://arxiv.org/html/2607.23838v1)
- 인용구: "CORDON-MAS reduces attack success rate by 92.4% relative to undefended RAG" (검증 수준: 검색 스니펫)

**D2. Ranking Is Not Deciding: Benchmark Shortcuts and Operating-Point Collapse in Query-Time RAG Poisoning Defenses**
- 저자: Pushkal Kumar et al. (Pushkal Kumar, Arvin Sahaym) — [arXiv 2610.33849](https://arxiv.org/abs/2610.33849)
- arXiv: 2610.33849 — https://arxiv.org/abs/2610.33849 · 최초 공개: 2026-10 (검색 시점 기준 약 26일 전 게시)
- 게재처: **Preprint (arXiv)**
- 요약(한국어):
  - 검색된 문서를 생성기 전에 걸러내는 query-time 오염 필터들이 도메인 벤치마크에서 거의 완벽해 보이는 이유가 필터의 능력이 아니라 벤치마크 오염문서 구성 방식의 지름길(질문 문장 그대로 포함, 길이 단서)에 있음을 4개 코퍼스·4개 공격 계열·10개 방어로 분석한다.
  - 기존 방어 평가 21건 중 8건이 이런 방식으로 만든 오염문서를 사용했고 12건은 실제 운용 지점의 오탐률을 보고하지 않았으며, 단서를 제거하면 랭킹(AUROC)은 일부 유지되나 실제 판정(운용 임계값)은 붕괴한다고 보고한다(예: 접두어 제거 시 RAGuard AUROC 0.953→0.758). — [arXiv 2610.33849](https://arxiv.org/abs/2610.33849); 수치는 [검색 요약](https://arxiv.org/abs/2610.33849)
- 인용구: "the ranking survives; the decision does not" (검증 수준: 검색 스니펫)

**[E] 프라이버시 (Privacy — Membership Inference / Private Datastore)**

**E1. Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation**
- 저자: Ali Naseh et al. (Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, Amir Houmansadr) — [arXiv 2502.00306](https://arxiv.org/abs/2502.00306)
- arXiv: 2502.00306 — https://arxiv.org/abs/2502.00306 · 최초 공개: 2025-02 (ID 기준)
- 게재처: **ACM CCS 2025** (CCS '25, Taipei) — [ACM DL DOI 10.1145/3719027.3744840](https://dl.acm.org/doi/pdf/10.1145/3719027.3744840)
- 요약(한국어):
  - 기존 RAG 멤버십 추론 공격이 탈옥이나 부자연스러운 쿼리에 의존해 쉽게 탐지·쿼리 재작성으로 무력화된다는 점을 지적하고, 대상 문서가 데이터스토어에 있을 때만 답할 수 있는 자연어 질문들을 던지는 Interrogation Attack(IA)을 제안한다.
  - 약 30개 쿼리만으로 은밀하게 추론에 성공하며, 기존 공격 대비 TPR@1%FPR 2배 향상, 문서당 추론 비용 $0.02 미만을 보고한다. — [arXiv 2502.00306](https://arxiv.org/html/2502.00306v2); [Hugging Face Papers](https://huggingface.co/papers/2502.00306)
- 인용구: "we present Interrogation Attack (IA), a membership inference technique targeting documents in the RAG datastore" (검증 수준: 검색 스니펫)

**E2. Differentially Private Datastore Generation for Retrieval-Augmented Inference**
- 저자: Abdelrahman Abouelenein et al. (Abdelrahman Abouelenein, Marwan Torki) — [arXiv 2606.01413](https://arxiv.org/abs/2606.01413)
- arXiv: 2606.01413 — https://arxiv.org/abs/2606.01413 · 최초 공개: **2026-05-31** (제출일; 목록 공지 2026-06-02)
- 게재처: **Preprint (arXiv)**
- 요약(한국어):
  - 검색 증강 추론용 데이터스토어에 형식적 차등프라이버시(DP)를 부여하기 위해, SimHash 기반 LSH로 데이터를 유사도 보존 버킷으로 나누고 버킷별 클래스 투표 히스토그램에 보정된 DP 노이즈를 더한 "DP 데이터스토어"를 공개하는 방법을 제안한다.
  - ε=5에서 평균 정확도 손실 약 2.6%, 멤버십 추론 공격 정확도 53.60%(무작위 수준 근접)를 보고한다(7개 텍스트 분류 데이터셋 기준 — 생성형 RAG가 아닌 분류형 retrieval-augmented inference라는 점에 유의). — [arXiv 2606.01413](https://arxiv.org/html/2606.01413v1)
- 인용구: "calibrated DP noise is added to the accumulated vote for each bucket" (검증 수준: 검색 스니펫, 초록이 아닌 본문 문장일 가능성 있음)

**[F] 평가 & 벤치마크 (Evaluation / Benchmarks)**

**F1. The Great Nugget Recall: Automating Fact Extraction and RAG Evaluation with Large Language Models** (AutoNuggetizer)
- 저자: Ronak Pradeep et al. (Ronak Pradeep, Nandan Thakur, Shivani Upadhyay, Daniel Campos, Nick Craswell, Jimmy Lin 등) — [arXiv 2504.15068](https://arxiv.org/abs/2504.15068v1)
- arXiv: 2504.15068 — https://arxiv.org/abs/2504.15068 · 최초 공개: 2025-04 (ID 기준)
- 게재처: **SIGIR 2025** (arXiv Comments "to appear in SIGIR 2025") — [arXiv 2504.15068](https://arxiv.org/abs/2504.15068v1) (full/short 구분 미확인)
- 요약(한국어):
  - TREC QA(2003)의 nugget 평가 방식을 LLM으로 자동화한 AutoNuggetizer 프레임워크(LLM이 nugget을 만들고 시스템 답변에 할당)를 제안하고, TREC 2024 RAG Track 런으로 검증한다.
  - 완전 자동 nugget 평가 점수와 인간 기반 변형 간 런(run) 수준에서 강한 일치를 보였고, nugget 생성은 수작업으로 두고 할당만 자동화하면 일치도가 더 높아진다(단, nugget recall만 평가). — [arXiv HTML](https://arxiv.org/html/2504.15068v1)
- 인용구: "strong agreement at the run level" (검증 수준: 검색 스니펫)

**F2. DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents**
- 저자: Mingxuan Du et al. (Mingxuan Du, Benfeng Xu, Chiwei Zhu, Licheng Zhang, Xiaorui Wang, Zhendong Mao) — [arXiv 2506.11763](https://arxiv.org/abs/2506.11763)
- arXiv: 2506.11763 — https://arxiv.org/abs/2506.11763 · 최초 공개: 2025-06 (ID 기준)
- 게재처: **ICLR 2026 (Poster)** — [iclr.cc 포스터 페이지](https://iclr.cc/virtual/2026/poster/10008065) (OpenReview 결정 페이지는 직접 열람하지 못함)
- 요약(한국어):
  - 딥리서치 에이전트를 체계적으로 평가할 벤치마크가 없다는 문제를 지적하고, 22개 분야 도메인 전문가가 만든 박사 수준 연구 과제 100개(중·영 각 50개)와, 보고서 품질 평가 RACE(포괄성·깊이·지시이행·가독성) 및 인용 신뢰도 평가 FACT 두 프레임워크를 제안한다.
  - Gemini/OpenAI/Grok/Perplexity 딥리서치 등을 평가해 딥리서치 에이전트가 검색 도구를 단 LLM보다 유효 인용 수에서 크게 앞섬을 보였다. — [arXiv PDF](https://arxiv.org/pdf/2506.11763); [프로젝트 페이지](https://deepresearch-bench.github.io)
- 인용구: "100 PhD-level research tasks, each meticulously crafted by domain experts across 22 distinct fields" (검증 수준: 검색 스니펫)
- 범위 메모: 딥리서치 "에이전트" 자체는 타 연구자 범위이나, 본 항목은 *출력물(보고서·인용)의 평가*라는 점에서 포함.

**F3. DREAM: Deep Research Evaluation with Agentic Metrics**
- 저자: AWS Agentic AI·Georgia Tech 팀 (제1저자 이름은 검색 결과로 확인하지 못함 → "et al."만 표기 불가, 보고서 작성 시 arXiv에서 확인 필요) — [arXiv 2602.18940](https://arxiv.org/abs/2602.18940)
- arXiv: 2602.18940 — https://arxiv.org/abs/2602.18940 · 최초 공개: **2026-02-21**
- 게재처: **Preprint (arXiv)**
- 요약(한국어):
  - 정적 평가기는 도구 사용 능력이 없어 사실 정확성과 시간적 유효성을 검증할 수 없고, 유창하고 인용이 잘 달린 보고서가 사실·추론 오류를 가리는 "Mirage of Synthesis" 문제가 있다고 지적하며, 평가 자체를 도구 호출 에이전트가 수행하는(쿼리 무관 지표 + 쿼리별 적응형 지표) 참조 없는(reference-free) 평가 프로토콜을 제안한다.
  - 기존 벤치마크보다 사실적·시간적 열화에 훨씬 민감하며, 오염 강도를 높여도 DeepResearch Bench의 FACT 인용 정확도는 ≈100%로 거의 변하지 않았다고 보고한다. — [arXiv HTML](https://arxiv.org/html/2602.18940v1); FACT 관련 수치는 [검색 요약](https://arxiv.org/html/2602.18940v1) 기준
- 인용구: "Mirage of Synthesis" (검증 수준: 검색 스니펫)

#### 추가 후보 (게재처 미확인 또는 정보 부족 — 보고서에서 "참고"로만 사용 권장)
- 지식 충돌 2026 프리프린트들: FIDES (2606.05644, 반사실적 문맥 편집으로 디코더 충실성을 검색 품질과 분리해 평가) — [arXiv](https://arxiv.org/pdf/2606.05644); ConflictRAG (2605.17301, 충돌 탐지 F1 약 88.7% 보고, 본문 수치 불일치 지적 있음) — [arXiv](https://arxiv.org/html/2605.17301); RAG-Stress (2610.11183, 단일 주장 편집에 대한 증거 의존도 측정) — [arXiv](https://arxiv.org/pdf/2610.11183); Faithfulness-QA (2604.25313, 반사실적 개체 치환 학습/평가 데이터셋) — [arXiv](https://arxiv.org/html/2604.25313v2); "Does RAG Know When Retrieval Is Wrong?" (2605.14473) — [arXiv](https://arxiv.org/pdf/2605.14473).
- 귀속: EARAL (EMNLP 2026 Main, `2026.emnlp-main.530`; 로짓 기반 생성-시점 답변·귀속 동시 생성) — [ACL Anthology](https://aclanthology.org/2026.emnlp-main.530/). arXiv ID는 찾지 못함(관련 선행 프리프린트로 LoDIT, 2508.08942가 있으나 동일 논문인지 미확인 — [arXiv](https://arxiv.org/pdf/2508.08942)).
- 평가: MINTQA (ACL 2026 Main Long, `2026.acl-long.18`; 신규·롱테일 지식 다중 홉 QA, 22개 LLM을 파라메트릭/직접 RAG/다중홉 RAG로 평가) — [ACL Anthology](https://aclanthology.org/2026.acl-long.18/). 최초 arXiv 공개 시점 미확인.
- 방어: SDAG (2602.04711, 문서 간 cross-attention을 막는 블록 희소 어텐션) — [arXiv](https://arxiv.org/abs/2602.04711); GuardRAG (2610.35071, 학습 불필요 사후 검색 그룹 탐지) — [arXiv](https://arxiv.org/abs/2610.35071); "Prevented is Not Protected" (2610.40742, ASR이 방어로 인한 실패를 가린다는 주장) — [arXiv](https://arxiv.org/html/2610.40742).
- 공격: Confundo (2602.06616, 전처리·리랭킹·패러프레이즈에도 살아남도록 학습된 오염문서) — [arXiv](https://arxiv.org/html/2602.06616v1).
- 딥리서치 평가: DEER (2512.17776) — [arXiv](https://arxiv.org/html/2512.17776v5); "Why Your Deep Research Agent Fails?" (2601.22984, 전체 연구 궤적의 환각 평가) — [arXiv](https://arxiv.org/pdf/2601.22984); DR³-Eval (2604.14683) — [arXiv](https://arxiv.org/pdf/2604.14683).
- 서베이(분류 체계 참고): "A Survey of Secure Retrieval-Augmented Generation" (2604.08304) — [arXiv](https://arxiv.org/pdf/2604.08304); "Security and Privacy in RAG: Architectures, Threats, Defenses, and Future Directions" (2606.25533, 검색·문맥 구성·생성 단계별 위협 분류, MIA·인덱스 추론·오염·그래디언트 유출 포함) — [arXiv](https://arxiv.org/html/2606.25533v1).

### Inferences
- 검증된 게재처를 보면 RAG 보안·프라이버시 논문은 ML 학회보다 보안 4대 학회(USENIX Security, CCS)와 WWW로, 충실성·인용 논문은 ACL 계열로, 평가 논문은 SIGIR·ICLR로 분산되는 경향이 있다(표본 14편 기준의 관찰이며 통계적 결론은 아님).
- 2026년 프리프린트 다수(Cordon-MAS, Ranking Is Not Deciding, DREAM, DP Datastore)는 아직 게재처가 없으며, 이들 중 상당수가 ICLR 2027 등 심사 중일 가능성이 있으나 확인할 수 없다.

### Gaps
- arXiv/ACL Anthology/OpenReview 직접 접속이 차단되어 초록 전문·Comments 필드·OpenReview 결정 페이지를 직접 읽지 못했다. 모든 인용구는 검색 스니펫 기반이다.
- 일부 논문의 정확한 v1 일자(2402.07867, 2504.13079, 2510.17853, 2504.15068, 2506.11763, 2605.26754, 2610.33849)는 월 단위로만 확인했다.
- DREAM의 제1저자 이름을 확인하지 못했다.
- NeurIPS 2025 / ICML 2026 / NeurIPS 2026 / IEEE S&P 2026 / NDSS 2026에서 채택된 이 분야 논문은 검색으로 확인하지 못했다(존재하지 않는다는 뜻이 아님). 특히 RobustRAG(인증된 강건성) 계열의 2026년 후속 연구와 게재처는 확인하지 못했다.
- SIGIR 2026의 RAG 평가 논문은 검색에 나타나지 않았다.

## Q2. 이 하위 분야는 어떻게 세분화되는가? (강건성/충돌 vs 충실성/귀속 vs 보안 공격·방어 vs 프라이버시 vs 평가/벤치마크)

### Takeaway
6개 하위 범주를 제안한다: [A] 강건성·지식 충돌, [B] 충실성·환각 탐지·인용/귀속, [C] 보안-공격(코퍼스 오염·간접 프롬프트 인젝션), [D] 보안-방어 및 방어 평가 재검토, [E] 프라이버시, [F] 평가·벤치마크(전통 RAG 평가 ↔ 딥리서치 보고서 평가). 2026년에는 [C]/[D]가 가장 활발하며, [D] 안에 "방어 벤치마크 자체의 타당성 비판"이라는 메타평가 흐름이 새로 생겼다.

### Cited Findings
- 2026 보안 서베이는 RAG 위협을 검색(retrieval)·문맥 구성(context construction)·생성(generation) 단계로 나누고, 멤버십 추론을 인덱스 추론·오염·그래디언트 유출과 나란히 하나의 공격 클래스로 다룬다 — [arXiv 2606.25533](https://arxiv.org/html/2606.25533v1)
- 지식 충돌은 선행 서베이(Xu et al., 2024)에서 context–memory, inter-context, intra-memory 충돌로 분류된다(검색 요약을 통한 간접 인용) — [검색 결과 맥락: arXiv 2510.12460](https://arxiv.org/html/2510.12460v1)
- 분류별 배정:
  - **[A] 강건성 & 지식 충돌**: A1 MADAM-RAG (COLM 2025) — [arXiv](https://arxiv.org/abs/2504.13079v2); 2026 후보 FIDES, ConflictRAG, RAG-Stress, Faithfulness-QA (모두 Preprint) — [FIDES](https://arxiv.org/pdf/2606.05644), [ConflictRAG](https://arxiv.org/html/2605.17301), [RAG-Stress](https://arxiv.org/pdf/2610.11183), [Faithfulness-QA](https://arxiv.org/html/2604.25313v2)
  - **[B] 충실성·환각·귀속**: B1 FRANQ (ACL 2026 Findings) — [ACL Anthology](https://aclanthology.org/2026.findings-acl.338/); B2 CiteGuard (ACL 2026 Main) — [ACL Anthology](https://aclanthology.org/2026.acl-long.282/); 참고 EARAL (EMNLP 2026 Main) — [ACL Anthology](https://aclanthology.org/2026.emnlp-main.530/)
  - **[C] 보안-공격**: C1 PoisonedRAG (USENIX Security 2025, 기초) — [USENIX](https://www.usenix.org/conference/usenixsecurity25/presentation/zou); C2 Overcoming the Retrieval Barrier (USENIX Security 2026) — [USENIX](https://www.usenix.org/conference/usenixsecurity26/presentation/chang); C3 Hidden-in-Plain-Text (WWW 2026) — [ACM DL](https://dl.acm.org/doi/10.1145/3774904.3792853); C4 InceptionRAG (CCS 2026) — [arXiv](https://arxiv.org/abs/2609.16818v1)
  - **[D] 보안-방어 & 방어 평가**: D1 Cordon-MAS (Preprint) — [arXiv](https://arxiv.org/abs/2605.26754); D2 Ranking Is Not Deciding (Preprint) — [arXiv](https://arxiv.org/abs/2610.33849); (C4의 HODOR 방어도 해당)
  - **[E] 프라이버시**: E1 Riddle Me This (CCS 2025) — [ACM DL](https://dl.acm.org/doi/pdf/10.1145/3719027.3744840); E2 DP Datastore (Preprint) — [arXiv](https://arxiv.org/abs/2606.01413)
  - **[F] 평가·벤치마크**: F1 Great Nugget Recall (SIGIR 2025) — [arXiv](https://arxiv.org/abs/2504.15068v1); F2 DeepResearch Bench (ICLR 2026 Poster) — [iclr.cc](https://iclr.cc/virtual/2026/poster/10008065); F3 DREAM (Preprint) — [arXiv](https://arxiv.org/abs/2602.18940)
- 범주 간 경계 사례: C3(Hidden-in-Plain-Text)는 "공격 벤치마크"로 [C]와 [F]에 걸치고, 완화책으로 "attribution-gated answering"을 평가해 [B]와도 연결된다 — [arXiv 2601.10923](https://arxiv.org/pdf/2601.10923)

### Inferences
- [A]와 [D]는 방법론적으로 수렴 중이다: MADAM-RAG(문서별 에이전트 토론), InceptionRAG의 HODOR(문서별 개별 응답 후 다수결), Cordon-MAS(원문 비노출 Synthesizer), SDAG(문서 간 어텐션 차단)는 모두 "문서 격리(per-document isolation)"라는 동일한 설계 원리를 공유한다. 오정보(우발적)와 오염(적대적)을 같은 메커니즘으로 다루려는 흐름으로 해석된다.
- [B]와 [F]도 수렴 중이다: 인용 정확도는 평가 지표(DeepResearch Bench FACT)이자 방어 수단(attribution-gated answering)으로 쓰이지만, DREAM은 "잘 인용된 거짓"을 인용 정렬만으로는 잡을 수 없다고 주장한다.

### Gaps
- 프라이버시 범주는 2026년에 공격보다 방어·서베이 위주로 보였으며(검색 기준), 2026년의 새로운 RAG 멤버십 추론 공격 중 게재처가 확인된 것은 찾지 못했다(이미지 기반 RAG 대상 ImageAuditor 2606.03354는 멀티모달이라 범위 밖이고 내용 미확인 — [arXiv](https://arxiv.org/pdf/2606.03354)).
- 인증된 강건성(certified robustness, RobustRAG 계열)의 2025–2026 후속 연구는 이번 검색에서 확인하지 못해 별도 하위 범주로 채우지 못했다.

## Q3. 2026년에 새롭게 등장한 연구 방향은 무엇인가? (에이전트형/검색 에이전트 RAG 공격, 딥리서치 출력 평가 등)

### Takeaway
2026년의 새 방향은 ① "검색 장벽"을 넘어 에이전트 워크플로까지 도달하는 간접 프롬프트 인젝션(데이터 유출 등 실제 행동 유발), ② 거짓 결론을 코퍼스에 쓰지 않고 생성 시점에 합성시키는 은밀한(compositional) 오염, ③ 웹 숨김 콘텐츠를 매개로 한 소셜 웹 인젝션 벤치마크, ④ 탐지가 아닌 정보흐름 제어·격리 기반 방어와 "방어 벤치마크가 지름길 때문에 과대평가되었다"는 메타평가 비판, ⑤ 짧은 QA에서 딥리서치 보고서로 이동한 평가(그리고 인용 정확도 지표의 포화 비판), ⑥ 데이터스토어에 대한 형식적 DP 보장이다.

### Cited Findings
- **에이전트까지 이어지는 IPI**: 오염된 이메일 1통으로 다중 에이전트 워크플로에서 GPT-4o가 SSH 키를 유출하게 만드는 데 80% 이상 성공; 기존 방어는 악성 텍스트의 검색을 막지 못함 (USENIX Security 2026) — [USENIX](https://www.usenix.org/conference/usenixsecurity26/presentation/chang-hongyan); [arXiv 2601.07072](https://arxiv.org/pdf/2601.07072)
- 같은 USENIX Security 2026에서 에이전트 도구 호출의 인과 귀속으로 IPI를 막는 AttriGuard가 방어 측으로 채택됨 — [USENIX PDF](https://www.usenix.org/system/files/usenixsecurity26-he-yu.pdf) (RAG보다 에이전트 범위에 가까움)
- **은밀한 합성형 오염**: 거짓 결론이 코퍼스에 한 번도 쓰이지 않고 생성 중에만 형성되어 명시적 독성 콘텐츠 탐지 방어를 우회 (CCS 2026) — [arXiv 2609.16818](https://arxiv.org/html/2609.16818v1)
- **웹 숨김 콘텐츠 매개 IPI 벤치마크**: 6,200개 소셜 스타일 페이지, 5종 숨김 캐리어 (WWW 2026) — [arXiv 2601.10923](https://arxiv.org/pdf/2601.10923)
- **탐지→제어로의 방어 패러다임 전환**: "monitoring-control gap"(모순을 탐지해도 오염 주장에 따라 행동) 지적과 IFC 기반 다중 에이전트 방어 — [arXiv 2605.26754](https://arxiv.org/html/2605.26754v1); 관련하여 "When Detection Does Not Guarantee Resistance: Reasoning and Poisoned Context in RAG" (2608.17153) — [arXiv](https://arxiv.org/pdf/2608.17153)
- **방어 평가의 메타 비판**: 21개 방어 평가 중 8개가 지름길 있는 오염문서 사용, 12개는 운용 지점 오탐률 미보고 — [arXiv 2610.33849](https://arxiv.org/abs/2610.33849); ASR이 방어로 인한 실패를 숨긴다는 주장과 matched-query 평가 프로토콜 — [arXiv 2610.40742](https://arxiv.org/html/2610.40742); 조직적 오염에 대한 admission-time 방어의 근본 한계 — [arXiv 2608.16044](https://arxiv.org/pdf/2608.16044)
- **딥리서치 출력 평가**: DeepResearch Bench가 ICLR 2026 포스터로 채택 — [iclr.cc](https://iclr.cc/virtual/2026/poster/10008065); 같은 ICLR 2026에 "Towards Personalized Deep Research: Benchmarks and Evaluations"도 게재 — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/43ec75c1c7de5eb82aa7616a277ba3f4-Abstract-Conference.html); DREAM은 인용 정렬만으로는 "그럴듯하고 잘 인용된 거짓"을 잡을 수 없으며 DeepResearch Bench의 FACT 인용 정확도가 오염 강도와 무관하게 ≈100%로 유지된다고 보고 — [arXiv 2602.18940](https://arxiv.org/html/2602.18940v1); DEER는 claim 단위 팩트체크와 claim 간 의존성 역추적을 평가 축으로 추가 — [arXiv 2512.17776](https://arxiv.org/html/2512.17776v5)
- **LLM-as-judge 평가 표준화**: 고정 예산·클러스터 인식 LLM-as-judge 표준과 400개 다중 홉 질문(CS/ML, 재료과학) 스트레스 테스트, 클러스터된 벤치마크가 진전을 과대평가할 수 있다는 주장 (Preprint) — [arXiv 2605.27789](https://arxiv.org/html/2605.27789); ACL 2026 RAG4Reports 워크숍의 LLM-judge 연구는 인용 충실성이 RAG 출력 평가의 가장 핵심 신호라고 보고 — [ACL Anthology](https://aclanthology.org/2026.rag4reports-1.9/)
- **형식적 프라이버시 보장**: LSH 기반 DP 데이터스토어, ε=5에서 정확도 손실 약 2.6%, MIA 정확도 53.60% — [arXiv 2606.01413](https://arxiv.org/html/2606.01413v1)

### Inferences
- 공격 측은 "검색되게 만들기(retrievability)"와 "탐지 회피(stealth)"를 분리·결합하는 방향으로 진화했고(C2 트리거/공격 조각 분리, C4 잠복 문서 쌍), 그 결과 문서 단위 필터형 방어의 전제가 약해졌다. 이것이 D1·D2 같은 구조적 격리 방어와 방어 평가 재검토가 2026년에 동시에 늘어난 배경으로 보인다.
- RAG 평가의 무게중심은 "정답 일치"에서 "주장 단위 검증 + 인용 + 시간적 유효성"으로 옮겨가고 있으며, 평가자 자체가 도구를 쓰는 에이전트가 되는 추세(DREAM)가 나타났다. 이는 [B] 충실성/귀속 연구와 [F] 평가 연구의 경계를 흐리게 한다.

### Gaps
- 검색 에이전트(search-agent)·딥리서치 에이전트를 직접 겨냥한 오염/인젝션 공격 중 게재처가 확인된 2026 논문은 이번 조사에서 찾지 못했다(일반 에이전트 IPI 논문들 — pikit 2609.36817, Scaling Adaptive Prompt Injection Attacks with Agent Swarms 2611.11861, Blocking at the Boundary 2610.05163 — 은 제목·스니펫만 확인, 내용·게재처 미확인 — [arXiv 2611.11861](https://arxiv.org/html/2611.11861), [arXiv 2610.05163](https://arxiv.org/pdf/2610.05163)).
- NDSS 2026 / IEEE S&P 2026의 RAG 관련 채택 논문은 확인하지 못했다.
- 2026년 신규 "신선도(freshness)" 특화 RAG 벤치마크 중 게재처가 확인된 것은 찾지 못했다(MINTQA가 신규·롱테일 지식을 다루는 가장 가까운 사례).
