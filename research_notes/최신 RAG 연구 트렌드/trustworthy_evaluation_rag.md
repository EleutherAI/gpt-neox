# 신뢰할 수 있는 RAG(Trustworthy RAG) 및 RAG 평가/벤치마크 — 최신 논문 동향 (기준일: 2026-11-19)

> **검증 방법에 관한 고지 (리포트 작성자 필독) — 2026-11-19 1차 출처 재검증판**
> - 이전 판은 arXiv·ACL Anthology·OpenReview 접속이 막혀 검색 스니펫에 의존했다. 이번 판에서는 **목록의 모든 arXiv ID(본 목록 14편 + 추가 후보·인용 논문 24편, 총 38건)의 `arxiv.org/abs/<id>` 페이지를 직접 내려받아** 제목·저자·제출 이력(v1 일자)·Comments·Journal-ref·DOI·초록을 대조했다. 필요한 경우 `arxiv.org/html/<id>` 본문도 열어 초록 밖의 수치를 확인했다.
> - ACL Anthology(`aclanthology.org`)의 개별 논문 페이지와 EMNLP 2026 Main/Findings 볼륨 목록을 직접 열어 Main/Findings 구분·Anthology ID·쪽수를 확인했다.
> - **직접 확인하지 못한 출처**: `openreview.net` 포럼 페이지와 `api.openreview.net/notes?id=…`는 브라우저 챌린지(`ChallengeRequiredError`, HTTP 403)를 반환했고, `api.openreview.net/notes/search`는 응답하지만 2025–2026 학회 기록은 들어 있지 않았으며, `api2.openreview.net`은 프록시가 차단했다. `usenix.org`, `iclr.cc`, `dblp.org`, `proceedings.mlr.press`도 이 세션의 프록시에서 차단되었다(CONNECT 403 / WebFetch ENOTFOUND). 따라서 **OpenReview 결정(ICLR/COLM/ICML)과 USENIX 프로그램 페이지는 1차 출처로 확인하지 못했고**, 해당 라벨은 arXiv Comments에 근거가 있을 때만 ✅로, 그렇지 않으면 ⚠️(Unverified)로 표기했다.
> - 각 논문 아래 "검증:" 줄에 확인 결과를 적었다. 요약 2줄은 **arXiv 초록(최신 버전)** 기준으로 다시 썼으며, 초록 밖 수치는 "(본문)"으로 따로 표시했다.

## Q1. 2025–2026년 RAG 신뢰성·평가 분야의 대표 논문은 무엇이며, 각각 어느 학회에 채택되었나?

### Takeaway
2025–2026년의 대표 흐름은 다음 세 가지다. (1) PoisonedRAG(USENIX Security 2025)를 기점으로 코퍼스 오염과 간접 프롬프트 인젝션 공격이 정교해졌고(CCS 2026, WWW 2026), 이에 맞서 "격리(isolation)" 기반 방어가 나왔다. (2) 인용·귀속(attribution)과 충실성(faithfulness) 검증이 정식화되었다(ACL 2026 Main/Findings, EMNLP 2026 Main). (3) RAG 평가는 "짧은 QA 정답률"에서 "nugget 기반 자동평가(SIGIR 2025)"와 "딥리서치 보고서 평가(ICML 2026의 DEER, 2026 프리프린트 DREAM 등)"로 옮겨갔다. 14편(기초 논문 1편 포함)은 모두 arXiv에 실재하며 제목·제1저자·ID가 일치한다. 다만 재검증 결과 게재처 라벨 일부가 바뀌었다. DP Datastore는 Preprint가 아니라 **ICPR 2026**이고, "Overcoming the Retrieval Barrier"의 USENIX Security 2026과 DeepResearch Bench의 ICLR 2026은 **1차 출처로 확인되지 않음(⚠️)**이다.

### Cited Findings

#### 선정 논문 목록 (서브카테고리 순)

**[A] 강건성 & 지식 충돌 (Robustness / Knowledge Conflict)**

**A1. Retrieval-Augmented Generation with Conflicting Evidence** (MADAM-RAG / RAMDocs)
- 저자: Han Wang et al. (Han Wang, Archiki Prasad, Elias Stengel-Eskin, Mohit Bansal) — [arXiv 2504.13079](https://arxiv.org/abs/2504.13079)
- arXiv: 2504.13079 — https://arxiv.org/abs/2504.13079 · 최초 공개: **2025-04-17** (v2: 2025-08-12)
- 게재처: **COLM 2025** (arXiv Comments: "COLM 2025, Data and Code: …") — [arXiv 2504.13079](https://arxiv.org/abs/2504.13079)
- 검증: ✅ arXiv abs 확인(제목·저자 4명·v1 일자·초록 일치) / venue 출처: arXiv Comments "COLM 2025" (OpenReview 결정 페이지는 챌린지로 열람 불가)
- 요약(한국어):
  - 모호한 질의, 출처 간 상충 정보, 노이즈·오정보를 지금까지는 따로따로 다뤄 왔다는 점을 지적한다. 이 셋을 동시에 담은 데이터셋 RAMDocs와, LLM 에이전트들이 여러 라운드 토론하고 집계자(aggregator)가 모호성이 해소된 개체별 답을 모으면서 오정보·노이즈는 버리는 다중 에이전트 기법 MADAM-RAG를 제안한다.
  - Llama3.3-70B-Instruct 기준으로 강한 RAG 기준선보다 AmbigDocs에서 최대 11.40%, FaithEval에서 최대 15.80%(절대) 향상했다. RAMDocs에서는 기존 기준선(Llama3.3-70B-Instruct EM 32.60)이 크게 고전했으며, 지지 증거와 오정보의 불균형이 커질수록 MADAM-RAG에도 상당한 격차가 남는다. — [arXiv 2504.13079](https://arxiv.org/abs/2504.13079)
- 인용구: "RAMDocs (Retrieval with Ambiguity and Misinformation in Documents)" (검증 수준: ✅ arXiv 초록 원문)

**[B] 충실성·환각 탐지·인용/귀속 (Faithfulness / Hallucination / Attribution)**

**B1. Faithfulness-Aware Uncertainty Quantification for Fact-Checking the Output of Retrieval Augmented Generation** (FRANQ)
- 저자: Ekaterina Fadeeva et al. (공저자 9명 추가: Rubashevskii, Piatrashyn, Vashurin, Dhuliawala, Shelmanov, Baldwin, Nakov, Sachan, Panov) — [arXiv 2505.21072](https://arxiv.org/abs/2505.21072)
- arXiv: 2505.21072 — https://arxiv.org/abs/2505.21072 · 최초 공개: **2025-05-27** (최신 v5: 2026-04-29)
- 게재처: **Findings of ACL 2026** (ACL Anthology ID `2026.findings-acl.338`, pp. 6814–6836; Anthology 제목 표기는 "…Retrieval-Augmented Generation") — [ACL Anthology](https://aclanthology.org/2026.findings-acl.338/)
- 검증: ✅ arXiv abs 확인(arXiv Comments는 비어 있음) / venue 출처: ACL Anthology 페이지(Volume "Findings of the Association for Computational Linguistics: ACL 2026", 저자 10명 일치)
- 요약(한국어):
  - 기존 환각 완화·탐지 방법은 사실성(factuality)과 검색 증거에 대한 충실성(faithfulness)을 섞어 버려서, 검색 문서가 뒷받침하지 않을 뿐 사실인 진술도 환각으로 잘못 표시한다. FRANQ는 진술이 검색 문맥에 충실한지에 따라 서로 다른 불확실성 정량화(UQ) 기법을 적용해 사실성을 추정한다.
  - 사실성과 충실성이 모두 주석된 장문 QA 데이터셋을 새로 만들었다(자동 라벨링 + 어려운 사례는 수작업 검증). 여러 데이터셋·과제·LLM에서 기존 방법보다 RAG 응답의 사실 오류를 더 정확히 탐지한다고 보고한다. — [arXiv 2505.21072](https://arxiv.org/abs/2505.21072); [ACL Anthology](https://aclanthology.org/2026.findings-acl.338/)
- 인용구: "FRANQ achieves more accurate detection of factual errors in RAG-generated responses compared to existing approaches" (검증 수준: ✅ arXiv 초록 원문. Anthology 게재본은 "…existing uncertainty quantification and hallucination detection approaches"로 끝남)

**B2. CiteGuard: Faithful Citation Attribution for LLMs via Retrieval-Augmented Validation**
- 저자: Yee Man Choi et al. (Yee Man Choi, Xuehang Guo, Yi R. Fung, Qingyun Wang) — [arXiv 2510.17853](https://arxiv.org/abs/2510.17853)
- arXiv: 2510.17853 — https://arxiv.org/abs/2510.17853 · 최초 공개: **2025-10-15** (최신 v4: 2026-04-13)
- 게재처: **ACL 2026 Main (Long)** (arXiv Comments "ACL 2026 Main Conference"; ACL Anthology ID `2026.acl-long.282`, pp. 6241–6257) — [ACL Anthology](https://aclanthology.org/2026.acl-long.282/)
- 검증: ✅ arXiv abs 확인 / venue 출처: arXiv Comments + ACL Anthology(Volume 1: Long Papers)
- 요약(한국어):
  - LLM이 과학 글쓰기를 돕는 상황에서 인용의 정확성·충실성이 문제인데, LLM-as-a-Judge만으로는 신뢰하기 어렵다고 본다. 그래서 인용 평가를 "LLM이 단 인용이 같은 문장에 인간 저자가 달았을 인용과 일치하는가"를 묻는 인용 귀속 정렬(citation attribution alignment) 문제로 재정의하고, 검색을 활용하는 에이전트 프레임워크 CiteGuard를 제안한다.
  - CiteME 벤치마크에서 이전 기준선보다 10%p 높은 최대 68.1% 정확도를 기록해 인간 성능(69.2%)에 근접했다. 대체 가능한 유효 인용도 찾아내며 도메인을 넘어선 일반화 능력도 보인다. — [arXiv 2510.17853](https://arxiv.org/abs/2510.17853); [ACL Anthology](https://aclanthology.org/2026.acl-long.282/)
- 인용구: "CiteGuard improves over the prior baseline by 10 percentage points and achieves up to 68.1% accuracy on the CiteME benchmark, approaching human performance (69.2%)" (검증 수준: ✅ arXiv v4 초록 = Anthology 초록)

**[C] 보안: 공격 (Security — Corpus Poisoning / Indirect Prompt Injection)**

**C1. [기초 논문 / Foundational] PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models**
- 저자: Wei Zou et al. (Wei Zou, Runpeng Geng, Binghui Wang, Jinyuan Jia) — [arXiv 2402.07867](https://arxiv.org/abs/2402.07867)
- arXiv: 2402.07867 — https://arxiv.org/abs/2402.07867 · 최초 공개: **2024-02-12** (v3: 2024-08-13)
- 게재처: **USENIX Security 2025** (arXiv Comments "To appear in USENIX Security Symposium 2025") — [arXiv 2402.07867](https://arxiv.org/abs/2402.07867); 참고: [USENIX 발표 페이지](https://www.usenix.org/conference/usenixsecurity25/presentation/zou) (이번 세션에서 접속 불가)
- 검증: ✅ arXiv abs 확인 / venue 출처: arXiv Comments. 이전 판의 "pp. 3827–3844"는 usenix.org 접속이 막혀 ⚠️ 미확인
- 요약(한국어):
  - RAG의 지식 DB를 새로운 공격 표면으로 보고, 소수의 악성 텍스트를 주입해 공격자가 고른 질문에 공격자가 고른 답이 나오게 하는 최초의 지식 오염(knowledge corruption) 공격을 최적화 문제로 정식화한다. 공격자 지식 수준(블랙박스/화이트박스)에 따라 두 가지 해법을 제시한다.
  - 수백만 건 규모의 지식 DB에 대상 질문당 악성 텍스트 5개만 넣어도 90% 공격 성공률을 달성했고, 평가한 여러 방어는 불충분했다. — [arXiv 2402.07867](https://arxiv.org/abs/2402.07867)
- 인용구: "PoisonedRAG could achieve a 90% attack success rate when injecting five malicious texts for each target question into a knowledge database with millions of texts" (검증 수준: ✅ arXiv 초록 원문)

**C2. Overcoming the Retrieval Barrier: Indirect Prompt Injection in the Wild for LLM Systems**
- 저자: Hongyan Chang et al. (Hongyan Chang, Ergute Bao, Xinjian Luo, Ting Yu; 소속은 이번에 확인하지 못함) — [arXiv 2601.07072](https://arxiv.org/abs/2601.07072)
- arXiv: 2601.07072 — https://arxiv.org/abs/2601.07072 · 최초 공개: **2026-01-11** (v1만 존재)
- 게재처: **Preprint (arXiv)**. USENIX Security 2026 채택은 **Unverified**: arXiv Comments·Journal-ref가 비어 있고, usenix.org는 이번 세션에서 프록시가 차단했다. 이전 판은 검색 스니펫으로 USENIX '26 발표 페이지(URL이 `/presentation/chang`과 `/presentation/chang-hongyan` 두 가지로 서로 달랐음)를 인용했다.
- 검증: ✅ arXiv abs 확인(제목·저자·일자·초록·수치 일치) / ⚠️ venue: USENIX Security 2026은 1차 출처 미확인. usenix.org 접근을 허용한 뒤 재확인 필요
- 요약(한국어):
  - 기존 간접 프롬프트 인젝션(IPI) 연구는 "악성 콘텐츠가 실제로 검색되게 만드는" 가장 어려운 단계를 피해 왔다. 이 논문은 악성 콘텐츠를 검색을 보장하는 트리거 조각과 임의의 공격 목표를 담는 공격 조각으로 나누고, 임베딩 모델 API 접근만으로 트리거를 만드는 블랙박스 공격 알고리즘을 제안한다.
  - 11개 벤치마크·8개 임베딩 모델에서 거의 100% 검색에 성공했고, OpenAI 임베딩 기준 대상 질의당 최저 $0.21이 들었다. 자연 질의 하의 RAG·에이전트 end-to-end 공격도 보였는데, 다중 에이전트 워크플로에서 오염 이메일 1통으로 GPT-4o가 SSH 키를 유출하게 만드는 데 80% 이상 성공했다. 평가한 방어들은 악성 텍스트의 검색을 막지 못했다. — [arXiv 2601.07072](https://arxiv.org/abs/2601.07072)
- 인용구: "achieves near-100% retrieval across 11 benchmarks and 8 embedding models" (검증 수준: ✅ arXiv 초록 원문)

**C3. Hidden-in-Plain-Text: A Benchmark for Social-Web Indirect Prompt Injection in RAG** (OpenRAG-Soc)
- 저자: Haoze Guo et al. (Haoze Guo, Ziqi Wei; 소속 미확인) — [arXiv 2601.10923](https://arxiv.org/abs/2601.10923)
- arXiv: 2601.10923 — https://arxiv.org/abs/2601.10923 · 최초 공개: **2026-01-16** (v2: 2026-01-20)
- 게재처: **WWW 2026 (The ACM Web Conference 2026)** (arXiv Comments "WWW 2026", arXiv DOI 필드 10.1145/3774904.3792853) — [arXiv 2601.10923](https://arxiv.org/abs/2601.10923); [ACM DL DOI](https://dl.acm.org/doi/10.1145/3774904.3792853) (트랙 구분 미확인)
- 검증: ✅ arXiv abs 확인(v1·v2 초록 동일) / venue 출처: arXiv Comments + arXiv DOI 필드. ACM DL 페이지 자체는 열람하지 않음
- 요약(한국어):
  - RAG가 웹의 사용자 생성 콘텐츠에 근거를 두면서, 수집 파이프라인을 통과해 살아남는 웹 고유 매체(carrier)를 통한 간접 프롬프트 인젝션·검색 오염이 커지고 있다. 이를 평가하려고 소셜 코퍼스, 교체 가능한 sparse/dense 검색기, 배포 가능한 완화책(HTML/Markdown 정화, 유니코드 정규화, 귀속 기반 응답 attribution-gated answering)을 묶은 재현 가능한 벤치마크 겸 실험 하네스 OpenRAG-Soc를 제안한다.
  - 수집부터 생성까지 end-to-end 평가를 표준화하며 응답 시점의 공격 성공, sparse/dense 검색기의 순위 변화, 효용, 지연을 보고해 매체·방어 간 동등 비교를 가능하게 한다. 본문에 따르면 코퍼스는 5종 매체(hidden span 1,330, off-screen CSS 1,280, alt text 1,280, ARIA 980, zero-width 1,330)에 걸친 6,200개 소셜형 페이지다. — [arXiv 2601.10923](https://arxiv.org/abs/2601.10923); 6,200·매체 수치는 [arXiv HTML v1 §3.1](https://arxiv.org/html/2601.10923v1)
- 인용구: "hidden spans, off-screen CSS, alt text, ARIA, zero-width" (검증 수준: ✅ arXiv 본문 §3·§3.1. 초록 문구는 아님)

**C4. InceptionRAG: Stealthy Poisoning Attack Against Retrieval-Augmented Generation**
- 저자: Jiachang Zhang et al. (Jiachang Zhang, Min Chen, Xiao Ren, Zhenyong Zhang, Yuanchao Shu, Yunjun Gao, Zhikun Zhang) — [arXiv 2609.16818](https://arxiv.org/abs/2609.16818)
- arXiv: 2609.16818 — https://arxiv.org/abs/2609.16818 · 최초 공개: **2026-09-15**
- 게재처: **ACM CCS 2026** (arXiv Comments: "20 pages, 5 figures. Accepted to appear in the Proceedings of the 2026 ACM SIGSAC Conference on Computer and Communications Security (CCS '26)") — [arXiv 2609.16818](https://arxiv.org/abs/2609.16818)
- 검증: ✅ arXiv abs 확인 / venue 출처: arXiv Comments(CCS '26 채택 명시)
- 요약(한국어):
  - 기존 RAG 오염 공격은 악성 페이로드를 문서 한 개에 통째로 담는 "단일 지점 명시적 주입"이어서 최근 완화책에 잡힌다. InceptionRAG는 페이로드를 각각은 무해해 보이는 "잠복(dormant) 문단의 사슬"로 쪼개, 함께 검색되었을 때 LLM이 다중 홉 추론으로 거짓 결론을 스스로 도출하게 만든다(간접 논리 유도). 블랙박스 환경에서는 권위 있어 보이는 접미사를 자동 생성하는 영차(zeroth-order) 접미사 최적화 ZOSO를 쓴다. 본문의 기본 구성은 대리 개체(proxy entity)를 매개로 한 속성 문서와 연결(bridge) 문서 2개다.
  - 3개 데이터셋 × 3개 LLM에서 엄격한 제약 아래서도 공격 성공률 80% 이상을 달성하고 단일 문서 주입용 기존 방어를 우회한다. 저자들은 "추론 능력이 강할수록 추론 기반 오염에 더 취약하다"는 역설을 지적하고, 적대적 논리 의존성을 끊는 문서 격리 방어 HODOR를 제안한다. 본문 기준으로 HODOR는 문서별 독립 응답 후 다수결로 동작하며, HotpotQA 같은 다중 홉 과제의 정상 성능을 눈에 띄게 떨어뜨린다. — [arXiv 2609.16818](https://arxiv.org/abs/2609.16818); HODOR 세부는 [arXiv HTML v1](https://arxiv.org/html/2609.16818v1)
- 인용구: "we propose a document isolation-based defense, HODOR, which decouples adversarial logical dependencies" (검증 수준: ✅ arXiv 초록 원문. 이전 판 인용구 문구를 원문대로 수정)

**[D] 보안: 방어 및 방어 평가의 재검토 (Defenses & Defense Meta-evaluation)**

**D1. Cordon-MAS: Defending RAG against Knowledge Poisoning via Information-Flow Control**
- 저자: Zhe Yu et al. (Zhe Yu, Wenpeng Xing, Gaolei Li, Shuguang Xiong, Hongzhi Wang, Xuyang Teng, Meng Han) — [arXiv 2605.26754](https://arxiv.org/abs/2605.26754) (arXiv 저자 목록이 이전 판 기재와 일치. "다른 저자 목록 인용" 우려는 해소됨)
- arXiv: 2605.26754 — https://arxiv.org/abs/2605.26754 · 최초 공개: **2026-05-26** (v1만 존재)
- 게재처: **Preprint (arXiv)** (Comments·Journal-ref 없음)
- 검증: ✅ arXiv abs 확인 / venue: 채택 정보 없음 → Preprint (arXiv)
- 요약(한국어):
  - 기존 방어는 "오염 증거를 탐지하면 피해를 막을 수 있다"고 가정하지만, 모델은 검색 증거의 모순을 탐지하고도 오염된 주장에 따라 행동한다(monitoring-control gap). 그래서 "최종 합성 능력이 있는 에이전트는 신뢰할 수 없는 자연어 증거에 접근할 수 없다"는 Cordon 원칙을 세우고, 증거 추출·출처 간 교차 감사·답변 합성을 비대칭 메모리 권한을 가진 에이전트로 나눠 이 원칙을 구조적으로 강제하는 CORDON-MAS를 제안한다(Confundo류 오염 대상).
  - 5개 BEIR 데이터셋에서 방어 없는 RAG 대비 공격 성공률을 92.4% 줄였으며, RAG 오염을 탐지 문제가 아닌 정보흐름 제어(IFC) 문제로 재정의한다. 본문 구성은 Extractor(원문 → 구조화된 claim card), Auditor, Gate(독립적인 2차 차단층), Synthesizer(인증된 claim만 사용)다. TriShieldRAG v2는 그 비용을 "질의당 3–4회 순차 LLM 호출"로 지적한다. — [arXiv 2605.26754](https://arxiv.org/abs/2605.26754); 구성은 [arXiv HTML v1](https://arxiv.org/html/2605.26754v1); 비용은 [TriShieldRAG v2 HTML](https://arxiv.org/html/2607.23838v2)
- 인용구: "CORDON-MAS reduces attack success rate by 92.4% relative to undefended RAG" (검증 수준: ✅ arXiv 초록 원문)

**D2. Ranking Is Not Deciding: Benchmark Shortcuts and Operating-Point Collapse in Query-Time RAG Poisoning Defenses**
- 저자: Pushkal Kumar et al. (Pushkal Kumar, Arvin Sahaym) — [arXiv 2610.33849](https://arxiv.org/abs/2610.33849)
- arXiv: 2610.33849 — https://arxiv.org/abs/2610.33849 · 최초 공개: **2026-10-24**
- 게재처: **Preprint (arXiv)** (Comments: "26 pages, 7 figures, 14 tables". 채택 정보 없음)
- 검증: ✅ arXiv abs 확인 / venue: 채택 정보 없음 → Preprint (arXiv)
- 요약(한국어):
  - 검색된 문서를 생성기 앞에서 하나씩 거르는 query-time 필터가 도메인 벤치마크에서 거의 완벽한 탐지를 보고하는 이유는 필터가 무엇을 읽느냐가 아니라 오염 세트의 구성 방식에 있다고 주장한다. 그 단서는 두 가지 무라벨 통계, 즉 containment(대상 질문이 오염문서에 그대로 들어간 비율)와 길이 중첩으로 프로파일링된다. PoisonedRAG 공개 코드는 모든 삽입 문단 앞에 대상 질문을 붙이며, 조사한 21개 방어 평가 중 8개가 이렇게 만든 오염문서를, 5개가 길이 단서를 사용했고, 12개는 실제 운용 임계값에서의 오탐률을 보고하지 않았다.
  - 접두 질문만 지우면 재구현한 RAGuard의 AUROC는 0.953→0.758, 미탐률은 0.191→0.763이 된다. 의료 오염문서를 내용 없는 단어 섞기 패딩으로 정상 길이에 맞추면 TrustRAG 집합 단계의 미탐률이 0.000→0.827, 저자들의 학습 점수기도 0.000→0.986이 되지만 AUROC는 0.953을 유지한다(4개 코퍼스·4개 공격 계열·10개 방어). 저자들은 자신들의 점수기도 방어 인지 공격자에게 뚫리므로 방어를 주장하지 않으며, 탐지 결과는 구성 프로파일과 운용 임계값 오탐률을 함께 보고해야 한다고 제안한다. — [arXiv 2610.33849](https://arxiv.org/abs/2610.33849)
- 인용구: "The ranking survives; the decision does not." (검증 수준: ✅ arXiv 초록 원문)

**[E] 프라이버시 (Privacy — Membership Inference / Private Datastore)**

**E1. Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation**
- 저자: Ali Naseh et al. (Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, Amir Houmansadr) — [arXiv 2502.00306](https://arxiv.org/abs/2502.00306)
- arXiv: 2502.00306 — https://arxiv.org/abs/2502.00306 · 최초 공개: **2025-02-01** (v2: 2025-06-30)
- 게재처: **ACM CCS 2025** (arXiv Comments: "This is the full version (27 pages) of the paper … published at CCS 2025") — [arXiv 2502.00306](https://arxiv.org/abs/2502.00306); 참고: [ACM DL DOI 10.1145/3719027.3744840](https://dl.acm.org/doi/pdf/10.1145/3719027.3744840) (DOI 페이지 자체는 열람하지 않음)
- 검증: ✅ arXiv abs 확인 / venue 출처: arXiv Comments
- 요약(한국어):
  - 기존 RAG 멤버십 추론·데이터 추출 공격은 탈옥이나 부자연스러운 쿼리에 의존해 탐지되거나 쿼리 재작성으로 쉽게 무력화된다. 이를 지적하며, 대상 문서가 데이터스토어에 있을 때만 답할 수 있는 자연어 질문을 만드는 Interrogation Attack(IA)을 제안한다.
  - 30개 쿼리만으로 은밀하게 추론에 성공했다. 단순 탐지기는 기존 방법의 적대적 프롬프트를 IA보다 최대 약 76배 더 자주 잡아냈고, 기존 공격 대비 TPR@1%FPR이 2배 향상되었으며, 문서당 추론 비용은 $0.02 미만이다. — [arXiv 2502.00306](https://arxiv.org/abs/2502.00306)
- 인용구: "we present Interrogation Attack (IA), a membership inference technique targeting documents in the RAG datastore" (검증 수준: ✅ arXiv 초록 원문)

**E2. Differentially Private Datastore Generation for Retrieval-Augmented Inference**
- 저자: Abdelrahman Abouelenein et al. (Abdelrahman Abouelenein, Marwan Torki) — [arXiv 2606.01413](https://arxiv.org/abs/2606.01413)
- arXiv: 2606.01413 — https://arxiv.org/abs/2606.01413 · 최초 공개: **2026-05-31** (제출 시각 19:18 UTC. 이전 판의 "목록 공지 2026-06-02"는 확인하지 못해 삭제)
- 게재처: **ICPR 2026** (arXiv Comments: "Accepted at the 28th International Conference on Pattern Recognition (ICPR-2026)") — [arXiv 2606.01413](https://arxiv.org/abs/2606.01413) ← **이전 판의 "Preprint (arXiv)"를 정정**
- 검증: ✅ arXiv abs 확인 / venue 출처: arXiv Comments(ICPR 2026 채택 명시)
- 요약(한국어):
  - 온디바이스 AI가 쓰는 검색 증강 추론용 데이터스토어를 개인정보 침해 없이 공개·공유하도록, 해시 기반 확률 생성 프레임워크를 제안한다. LSH로 고차원 데이터를 버킷으로 나누고 버킷별 누적 클래스 투표에 보정된 DP 노이즈를 더해 클래스 확률 분포를 만든다. 본문에 따르면 SimHash(무작위 초평면 해싱)를 쓰고 KNN-Prompting 데이터스토어를 변환한다.
  - 클래스 수 2–14개의 7개 데이터셋(본문: MR, SUBJ, TREC, CR, AGNews, DBpedia, MPQA 텍스트 분류)에서 ε=5일 때 평균 정확도 손실 2.6%, 멤버십 추론 공격 정확도 53.60%를 보고한다. 생성형 RAG가 아니라 분류형 검색 증강 추론이라는 점에 유의해야 한다. — [arXiv 2606.01413](https://arxiv.org/abs/2606.01413); 데이터셋·SimHash는 [arXiv HTML v1](https://arxiv.org/html/2606.01413v1)
- 인용구: "We then add calibrated DP noise to the accumulated vote for each bucket" (검증 수준: ✅ arXiv 초록 원문. 이전 판의 "본문 문장일 가능성" 우려는 해소됨)

**[F] 평가 & 벤치마크 (Evaluation / Benchmarks)**

**F1. The Great Nugget Recall: Automating Fact Extraction and RAG Evaluation with Large Language Models** (AutoNuggetizer)
- 저자: Ronak Pradeep et al. (Ronak Pradeep, Nandan Thakur, Shivani Upadhyay, Daniel Campos, Nick Craswell, Jimmy Lin. 총 6명) — [arXiv 2504.15068](https://arxiv.org/abs/2504.15068)
- arXiv: 2504.15068 — https://arxiv.org/abs/2504.15068 · 최초 공개: **2025-04-21** (v1만 존재)
- 게재처: **SIGIR 2025** (arXiv Comments: "To appear in SIGIR 2025. Significant updates and revisions to arXiv:2411.09607") — [arXiv 2504.15068](https://arxiv.org/abs/2504.15068) (full/short 구분 미확인)
- 검증: ✅ arXiv abs 확인 / venue 출처: arXiv Comments
- 요약(한국어):
  - TREC QA Track(2003)의 nugget 평가(좋은 답에 있어야 할 원자적 사실 기준)를 LLM으로 "리팩터링"해, nugget 생성과 시스템 답변에 대한 nugget 할당을 모두 자동화한 AutoNuggetizer 프레임워크를 제안한다. TREC 2024 RAG Track에서 사람이 수작업·반수작업으로 nugget을 만들고 할당한 방식과 비교해 보정했다.
  - 완전 자동 nugget 평가와 인간 기반 변형 사이에 런(run) 수준의 강한 일치를 보였고, nugget 할당 같은 개별 구성요소를 따로 자동화하면 일치도가 더 높았다. 다만 시스템 실패 진단에 필요한 주제(topic) 수준의 일치는 추가 연구가 필요하다. — [arXiv 2504.15068](https://arxiv.org/abs/2504.15068)
- 인용구: "we observe strong agreement at the run level between scores derived from fully automatic nugget evaluation and human-based variants" (검증 수준: ✅ arXiv 초록 원문)

**F2. DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents**
- 저자: Mingxuan Du et al. (Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, Zhendong Mao. arXiv 기준 5명이며, **이전 판의 "Licheng Zhang"은 arXiv 저자 목록에 없어 삭제**) — [arXiv 2506.11763](https://arxiv.org/abs/2506.11763)
- arXiv: 2506.11763 — https://arxiv.org/abs/2506.11763 · 최초 공개: **2025-06-13** (v1만 존재)
- 게재처: **ICLR 2026 (Poster) — Unverified (1차 출처 미확인)**. arXiv Comments는 "31 pages, 5 figures"뿐이다. iclr.cc는 프록시가 차단했고, OpenReview 포럼(검색 결과에 나온 id `hQ0K2Hhq7H`)은 브라우저 챌린지로 열리지 않았으며, `api.openreview.net` 검색에는 2026 기록이 없다. 검색 엔진에는 [iclr.cc/virtual/2026/poster/10008065](https://iclr.cc/virtual/2026/poster/10008065)와 mlanthology ICLR 2026 항목이 같은 제목으로 색인되어 있다(스니펫 수준).
- 검증: ✅ arXiv abs 확인(저자 수 정정) / ⚠️ venue: ICLR 2026 Poster는 검색 색인으로만 확인. OpenReview 결정은 미열람
- 요약(한국어):
  - 딥리서치 에이전트(DRA)를 체계적으로 평가할 벤치마크가 없다는 문제를 지적하고, 22개 분야 도메인 전문가가 만든 박사 수준 연구 과제 100개로 이루어진 DeepResearch Bench를 제안한다(본문: 중국어 50개, 영어 50개).
  - 인간 판단과 잘 맞는 두 평가 방법을 제안한다. 하나는 적응형 기준을 쓰는 참조 기반 보고서 품질 평가(본문 명칭 RACE)이고, 다른 하나는 유효 인용 수와 전체 인용 정확도로 정보 검색·수집 능력을 재는 프레임워크(본문 명칭 FACT)다. 본문 결과로는 Grok을 제외한 딥리서치 에이전트들이 검색 도구를 단 LLM보다 유효 인용을 더 많이 포함했고, Gemini-2.5-Pro Deep Research가 평균 111.21개로 가장 많았다. — [arXiv 2506.11763](https://arxiv.org/abs/2506.11763); 본문 수치는 [arXiv HTML v1](https://arxiv.org/html/2506.11763v1)
- 인용구: "a benchmark consisting of 100 PhD-level research tasks, each meticulously crafted by domain experts across 22 distinct fields" (검증 수준: ✅ arXiv 초록 원문)
- 범위 메모: 딥리서치 "에이전트" 자체는 다른 연구자의 범위지만, 이 항목은 *출력물(보고서·인용)의 평가*를 다루므로 포함했다.

**F3. DREAM: Deep Research Evaluation with Agentic Metrics**
- 저자: **Elad Ben Avraham et al.** (Elad Ben Avraham, Changhao Li, Ron Dorfman, Roy Ganz, Oren Nuriel, Amir Dudai, Aviad Aberdam, Noah Flynn, Elman Mansimov, Adi Kalyanpur, Ron Litman. 총 11명. 본문 기준 소속은 AWS Agentic AI이며, Changhao Li는 Georgia Institute of Technology 소속으로 AWS 인턴 중 수행) — [arXiv 2602.18940](https://arxiv.org/abs/2602.18940) ← **이전 판에서 미상이던 제1저자 확인**
- arXiv: 2602.18940 — https://arxiv.org/abs/2602.18940 · 최초 공개: **2026-02-21**
- 게재처: **Preprint (arXiv)** (Comments·Journal-ref 없음)
- 검증: ✅ arXiv abs 확인 / venue: 채택 정보 없음 → Preprint (arXiv)
- 요약(한국어):
  - 기존 딥리서치 벤치마크는 표면적 유창성과 인용 정렬이 사실·추론 결함을 가리는 "Mirage of Synthesis" 문제를 안고 있다. 4개 축(vertical)의 분류 체계로 이 간극을 분석한 결과, 정적 평가기는 도구 사용 능력이 없어 시간적 유효성과 사실 정확성을 검증할 수 없다는 능력 불일치가 드러났다. 이를 해결하려고 평가 자체를 에이전트화하는 "능력 동등(capability parity)" 원칙의 DREAM을 제안한다.
  - DREAM은 쿼리 무관 지표와 도구 호출 에이전트가 생성하는 적응형 지표를 결합해 시간 인식 커버리지, 근거 기반 검증, 체계적 추론 탐침을 제공하는 참조 없는(reference-free) 평가 패러다임이다. 통제 실험에서 기존 벤치마크보다 사실적·시간적 열화에 훨씬 민감했다. 본문 실험에서는 오염 비율을 높여도 DeepResearch Bench의 FACT 인용 정확도가 사실상 일정하게 유지되었다(정확한 값은 수식 렌더링이라 추출하지 못했으므로 이전 판의 "≈100%"는 보류). — [arXiv 2602.18940](https://arxiv.org/abs/2602.18940); FACT 실험은 [arXiv HTML v1](https://arxiv.org/html/2602.18940v1)
- 인용구: "they suffer from the Mirage of Synthesis, where strong surface-level fluency and citation alignment can obscure underlying factual and reasoning defects" (검증 수준: ✅ arXiv 초록 원문)

#### 추가 후보 (보고서에서 "참고"로 사용 권장) — 모두 arXiv abs 또는 ACL Anthology에서 실재 확인
- 지식 충돌 관련 2026 프리프린트:
  - **FIDES** (2606.05644, Zhe Yu et al., v1 2026-06-04, Preprint): 검색 증거와 파라메트릭 기억 사이의 충돌이 소수의 답변 핵심 토큰에 집중된다는 점(token-level conflict concentration)을 이용한다. 세 가지 내부 신호(출력 표면, 은닉 표현, 예측 궤적)로 디코딩 단계마다 대조 강도를 조절하는 **학습 불필요 디코더**다. 3개 벤치마크 × 6개 백본의 18개 설정 모두에서 문맥 충실도 최고를 기록했고, 가장 강한 학습 불필요 기준선보다 +3~+13점 높았다. — [arXiv](https://arxiv.org/abs/2606.05644) (검증: ✅. **이전 판의 "반사실적 문맥 편집으로 디코더 충실성을 검색 품질과 분리해 평가"라는 설명은 초록과 맞지 않아 정정**)
  - **ConflictRAG** (2605.17301, Chenyu Wang et al., v1 2026-05-17): 2단계 충돌 탐지(탐지 정확도 90.8%, API 비용 62% 절감), Entropy-TOPSIS 출처 신뢰도 평가, CARS 지표를 제안했고, 충돌 탐지 F1 88.7%를 보고한다. Comments: "submitted to IEEE SMC 2026" → Preprint (arXiv). — [arXiv](https://arxiv.org/abs/2605.17301) (검증: ✅. 이전 판의 "본문 수치 불일치 지적 있음"은 근거를 확인하지 못해 삭제)
  - **RAG-Stress** (2610.11183, Shunyuan Zhou et al., v1 2026-10-08, Preprint): 주장 1개를 편집해 오답을 지지하게 만들고 오도율(MR)을 측정한다. 15개 시스템을 평가했으며, 문서 우선 지시가 사전지식 허용 지시보다 MR이 10.9–13.5%p 높았다. — [arXiv](https://arxiv.org/abs/2610.11183) (검증: ✅)
  - **Faithfulness-QA** (2604.25313, Li Ju et al., v1 2026-04-28, Preprint): SQuAD·TriviaQA를 바탕으로 개체를 반사실적으로 치환한 99,094개 샘플(개체 은행 76,953개)로, 학습용과 평가용을 겸한다. — [arXiv](https://arxiv.org/abs/2604.25313) (검증: ✅)
  - **"Does RAG Know When Retrieval Is Wrong? Diagnosing Context Compliance under Knowledge Conflict"** (2605.14473, Yihang Chen et al., v1 2026-05-14, Comments "Preprint"): 진단 개입 기법 CDD를 제안하며, 최악의 TruthfulQA 오개념 주입에서 Standard RAG 정확도가 15.0%였다. — [arXiv](https://arxiv.org/abs/2605.14473) (검증: ✅)
- 귀속: **EARAL: Efficient Attribution of Retrieval-Augmented Answer Generation using Logits** (EMNLP 2026 Main, `2026.emnlp-main.530`, pp. 12233–12247; Lucas Albarede, Jose G. Moreno, Lynda Tamine, Luce Lefeuvre) — [ACL Anthology](https://aclanthology.org/2026.emnlp-main.530/). **arXiv 프리프린트는 2508.08942** ("Jointly Generating and Attributing Answers using Logits of Document-Identifier Tokens", 방법명 LoDIT, v1 2025-08-12)로 확인되었다. arXiv DOI 필드가 10.18653/2026.emnlp-main.530을 가리키고 저자 4명과 방법(문서 식별자 토큰 로짓으로 생성 중 귀속, Trust-Align 평가)이 같다. — [arXiv](https://arxiv.org/abs/2508.08942) (검증: ✅ 이전 판의 "동일 논문인지 미확인"을 해소)
- 평가: **MINTQA** (ACL 2026 Main Long, `2026.acl-long.18`, pp. 445–479; 제1저자 Jie He): 신규/기존 지식 10,479쌍과 인기/비인기 지식 17,887쌍의 다중 홉 QA다. 22개 LLM을 파라메트릭 QA, 직접 RAG, 다중 홉 RAG로 평가했다. — [ACL Anthology](https://aclanthology.org/2026.acl-long.18/) (검증: ✅ Anthology 확인. arXiv ID와 최초 공개 시점은 조사하지 않음)
- 방어:
  - **SDAG** — "Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention" (2602.04711, Sagie Dekel, Moshe Tennenholtz, Oren Kurland; v1 2026-02-04): 문서 간 cross-attention을 막는 블록 희소 어텐션이다. **게재처: EMNLP 2026 Main** (arXiv Journal-ref "Proceedings of EMNLP 2026, pp. 31952-31973"; ACL Anthology `2026.emnlp-main.1399`). — [arXiv](https://arxiv.org/abs/2602.04711) (검증: ✅. 이전 판에는 게재처 표기가 없었음)
  - **GuardRAG** (2610.35071, Linghui Jiao et al., v1 2026-10-26, Preprint): 학습 불필요 사후 검색 방어로, HAC 군집화를 써서 의심스러운 문서 그룹을 가지치기한다. — [arXiv](https://arxiv.org/abs/2610.35071) (검증: ✅)
  - **"Prevented is Not Protected: What Attack Success Rate Hides about RAG"** (2610.40742, Chahana Dahal, Zuobin Xiong; v1 2026-10-29, Preprint): 정상/오염 × 방어 유무를 짝지은 matched-query 프로토콜(Aligned RAG Evaluation)이다. 방어가 오히려 대상 긍정 답을 유발할 수 있고, 평가 기준에 따라 방어 순위가 바뀐다. — [arXiv](https://arxiv.org/abs/2610.40742) (검증: ✅)
- 공격: **Confundo** (2602.06616, Haoyang Hu et al., v1 2026-02-06, Preprint): LLM을 오염문서 생성기로 미세조정하는 learning-to-poison 프레임워크다. 실제 시스템의 콘텐츠 전처리(오염문서 단편화)와 공격 설계 때 예상과 다른 사용자 질의에도 효과를 유지하며, 방어가 있어도 기존 공격을 크게 앞선다. — [arXiv](https://arxiv.org/abs/2602.06616) (검증: ✅. 이전 판의 "리랭킹·패러프레이즈"는 초록 표현이 아니라 "전처리·질의 변이"로 정정)
- 딥리서치 평가:
  - **DEER** (2512.17776, Janghoon Han et al., v1 2025-12-19): 전문가 분류 체계(7개 차원, 25개 하위 차원, 101개 루브릭 항목), 과제별 Expert Evaluation Guidance, 인용된/인용되지 않은 주장을 모두 검증하는 claim 검증 구조를 갖췄다. **게재처: ICML 2026** (arXiv Comments "ICML 2026 (45 pages, …)"). — [arXiv](https://arxiv.org/abs/2512.17776) (검증: ✅. 이전 판의 "claim 간 의존성 역추적"은 초록에 없어 정정했고, 게재처를 추가함. OpenReview/PMLR 페이지는 열람하지 못함)
  - **"Why Your Deep Research Agent Fails? On Hallucination Evaluation in Full Research Trajectory"** (2601.22984, Yuhao Zhan et al., v1 2026-01-30, Preprint): PING 분류 체계와 DeepHalluBench(100개 과제)로 6개 DRA를 평가한다. — [arXiv](https://arxiv.org/abs/2601.22984) (검증: ✅)
  - **DR³-Eval** (2604.14683, Qianqian Xie et al., v1 2026-04-16, Preprint): 과제별 정적 샌드박스 코퍼스와 5개 차원 평가로 이루어진 멀티모달·다중 파일 보고서 평가다. — [arXiv](https://arxiv.org/abs/2604.14683) (검증: ✅)
- 서베이(분류 체계 참고):
  - **"A Survey of Secure Retrieval-Augmented Generation"** (2604.08304, Yuming Xu et al., v1 2026-04-09): 135편을 SLOT 분류 체계(공격 표면 S, 방어 계층 L, 목적 O, 대상 범위 T)로 정리했다. **게재처: Findings of EMNLP 2026** (ACL Anthology `2026.findings-emnlp.241`. arXiv Comments에는 "Accepted at EMNLP 2026"이라고만 적혀 있으나 Anthology상 Findings) — [arXiv](https://arxiv.org/abs/2604.08304) (검증: ✅)
  - **"Security and Privacy in Retrieval-Augmented Generation: Architectures, Threats, Defenses, and Future Directions for Building Trustworthy Systems"** (2606.25533, Balamurugan Palanisamy et al., v1 2026-06-24, Preprint): 검색·문맥 구성·생성 단계별 위협 분류와 MIA·인덱스 추론·오염·그래디언트 유출·공모 공격 클래스를 다룬다. — [arXiv](https://arxiv.org/abs/2606.25533) (검증: ✅)

### Inferences
- 검증된 게재처를 보면 RAG 보안·프라이버시 논문은 ML 학회보다 보안 학회(USENIX Security, CCS)와 WWW로, 충실성·인용 논문은 ACL 계열(ACL 2026 Main/Findings, EMNLP 2026 Main)로, 평가 논문은 SIGIR·ICML(DEER)로 흩어지는 경향이 있다. 표본이 작아 통계적 결론은 아니다. 프라이버시 방어 중 하나(DP Datastore)는 패턴인식 학회(ICPR 2026)에 실렸다.
- 2026년 프리프린트 중 Cordon-MAS, Ranking Is Not Deciding, DREAM은 아직 arXiv 상 게재처 표기가 없다. 심사 중인지는 확인할 수 없다.

### Gaps
- OpenReview(포럼·API) 접근이 브라우저 챌린지와 프록시에 막혀 ICLR 2026(DeepResearch Bench), COLM 2025(MADAM-RAG), ICML 2026(DEER)의 공식 결정을 OpenReview에서 직접 확인하지 못했다. COLM·ICML은 arXiv Comments에 근거가 있고, ICLR 2026은 검색 색인뿐이다.
- usenix.org가 프록시에 막혀 "Overcoming the Retrieval Barrier"(2601.07072)의 USENIX Security 2026 채택과 PoisonedRAG의 쪽수를 확인하지 못했다.
- dl.acm.org는 시도하지 않았다(허용 호스트 목록 밖). WWW 2026 트랙과 CCS 2025 DOI 페이지는 미확인이다.
- NeurIPS 2025 / ICML 2026 / NeurIPS 2026 / IEEE S&P 2026 / NDSS 2026에 채택된 이 분야 논문은 체계적으로 조사하지 못했다(존재하지 않는다는 뜻이 아님). RobustRAG(인증된 강건성) 계열의 2026년 후속 연구도 확인하지 못했다.
- SIGIR 2026의 RAG 평가 논문은 조사하지 않았다.

## Q2. 이 하위 분야는 어떻게 세분화되는가? (강건성/충돌 vs 충실성/귀속 vs 보안 공격·방어 vs 프라이버시 vs 평가/벤치마크)

### Takeaway
6개 하위 범주를 제안한다: [A] 강건성·지식 충돌, [B] 충실성·환각 탐지·인용/귀속, [C] 보안-공격(코퍼스 오염·간접 프롬프트 인젝션), [D] 보안-방어 및 방어 평가 재검토, [E] 프라이버시, [F] 평가·벤치마크(전통 RAG 평가와 딥리서치 보고서 평가). 2026년에는 [C]/[D]가 가장 활발하며, [D] 안에서 "방어 벤치마크 자체의 타당성 비판"이라는 메타평가 흐름이 새로 생겼다.

### Cited Findings
- 2026 보안·프라이버시 서베이는 RAG 위협 표면을 검색(retrieval)·문맥 구성(context construction)·생성(generation) 단계로 나누는 통합 분류 체계를 제시한다. 멤버십 추론, 인덱스 추론, 오염, 그래디언트 유출, 공모(collusion)를 공격 클래스로 분석한다. — [arXiv 2606.25533](https://arxiv.org/abs/2606.25533) (✅ 초록)
- EMNLP 2026 Findings 서베이는 135편을 SLOT(공격 표면·방어 계층·목적·대상 범위)으로 정리하고 세 가지 불일치(target/stage/signal mismatch)를 지적한다. — [arXiv 2604.08304](https://arxiv.org/abs/2604.08304); [ACL Anthology 볼륨 목록](https://aclanthology.org/volumes/2026.findings-emnlp/) (✅)
- 지식 충돌은 선행 서베이(Xu et al., 2024)에서 context–memory, inter-context, intra-memory 충돌로 분류된다. 이 문장은 이전 판에서 2510.12460(ProbeRAG, ACL 2026 Findings: arXiv Comments) 맥락으로 간접 인용한 것이며, 해당 분류 문장은 이번에 본문에서 재확인하지 않았다(⚠️). — [arXiv 2510.12460](https://arxiv.org/abs/2510.12460)
- 분류별 배정(검증된 게재처 라벨):
  - **[A] 강건성 & 지식 충돌**: A1 MADAM-RAG (COLM 2025) — [arXiv](https://arxiv.org/abs/2504.13079); 2026 후보 FIDES, ConflictRAG, RAG-Stress, Faithfulness-QA, Context Compliance(2605.14473) (모두 Preprint) — [FIDES](https://arxiv.org/abs/2606.05644), [ConflictRAG](https://arxiv.org/abs/2605.17301), [RAG-Stress](https://arxiv.org/abs/2610.11183), [Faithfulness-QA](https://arxiv.org/abs/2604.25313)
  - **[B] 충실성·환각·귀속**: B1 FRANQ (Findings of ACL 2026) — [ACL Anthology](https://aclanthology.org/2026.findings-acl.338/); B2 CiteGuard (ACL 2026 Main) — [ACL Anthology](https://aclanthology.org/2026.acl-long.282/); 참고 EARAL/LoDIT (EMNLP 2026 Main) — [ACL Anthology](https://aclanthology.org/2026.emnlp-main.530/); ProbeRAG (ACL 2026 Findings, arXiv Comments) — [arXiv](https://arxiv.org/abs/2510.12460)
  - **[C] 보안-공격**: C1 PoisonedRAG (USENIX Security 2025, 기초) — [arXiv](https://arxiv.org/abs/2402.07867); C2 Overcoming the Retrieval Barrier (Preprint; USENIX Security 2026은 ⚠️ Unverified) — [arXiv](https://arxiv.org/abs/2601.07072); C3 Hidden-in-Plain-Text (WWW 2026) — [arXiv](https://arxiv.org/abs/2601.10923); C4 InceptionRAG (CCS 2026) — [arXiv](https://arxiv.org/abs/2609.16818)
  - **[D] 보안-방어 & 방어 평가**: D1 Cordon-MAS (Preprint) — [arXiv](https://arxiv.org/abs/2605.26754); D2 Ranking Is Not Deciding (Preprint) — [arXiv](https://arxiv.org/abs/2610.33849); SDAG (EMNLP 2026 Main) — [arXiv](https://arxiv.org/abs/2602.04711); (C4의 HODOR 방어도 해당)
  - **[E] 프라이버시**: E1 Riddle Me This (CCS 2025) — [arXiv](https://arxiv.org/abs/2502.00306); E2 DP Datastore (ICPR 2026) — [arXiv](https://arxiv.org/abs/2606.01413)
  - **[F] 평가·벤치마크**: F1 Great Nugget Recall (SIGIR 2025) — [arXiv](https://arxiv.org/abs/2504.15068); F2 DeepResearch Bench (ICLR 2026 Poster, ⚠️ 1차 미확인) — [arXiv](https://arxiv.org/abs/2506.11763); F3 DREAM (Preprint) — [arXiv](https://arxiv.org/abs/2602.18940); 참고 DEER (ICML 2026) — [arXiv](https://arxiv.org/abs/2512.17776)
- 범주 간 경계 사례: C3(Hidden-in-Plain-Text)는 "공격 벤치마크"라서 [C]와 [F]에 걸치고, 완화책으로 attribution-gated answering을 포함하므로 [B]와도 연결된다. — [arXiv 2601.10923](https://arxiv.org/abs/2601.10923)

### Inferences
- [A]와 [D]는 방법론적으로 수렴하고 있다. MADAM-RAG(문서별 에이전트 토론), InceptionRAG의 HODOR(문서별 개별 응답 후 다수결), Cordon-MAS(원문을 보지 않는 Synthesizer), SDAG(문서 간 어텐션 차단)는 모두 "문서 격리(per-document isolation)"라는 같은 설계 원리를 공유한다. 우발적 오정보와 적대적 오염을 같은 메커니즘으로 다루려는 흐름으로 해석된다. 다만 HODOR 본문은 이 격리가 다중 홉 과제의 정상 성능을 떨어뜨린다고 보고하므로, 격리와 다중 홉 추론 사이에 트레이드오프가 있다.
- [B]와 [F]도 수렴하고 있다. 인용 정확도는 평가 지표(DeepResearch Bench의 FACT)이면서 방어 수단(attribution-gated answering)으로도 쓰인다. 하지만 DREAM은 인용 정렬만으로는 "잘 인용된 거짓"을 잡을 수 없다고 주장한다.

### Gaps
- 프라이버시 범주에서 2026년의 새로운 텍스트 RAG 멤버십 추론 공격 중 게재처가 확인된 것은 찾지 못했다. 이미지 기반 RAG 대상 ImageAuditor(2606.03354, Jinghuai Zhang et al., v1 2026-06-02, Preprint)는 "IRAG에 맞춘 최초의 MIA"로 실재가 확인되었지만, 멀티모달이라 범위 밖이다. — [arXiv](https://arxiv.org/abs/2606.03354)
- 인증된 강건성(RobustRAG 계열)의 2025–2026 후속 연구는 조사하지 못해 별도 하위 범주로 채우지 못했다.

## Q3. 2026년에 새롭게 등장한 연구 방향은 무엇인가? (에이전트형/검색 에이전트 RAG 공격, 딥리서치 출력 평가 등)

### Takeaway
2026년의 새 방향은 여섯 가지다. ① "검색 장벽"을 넘어 에이전트 워크플로까지 닿는 간접 프롬프트 인젝션(데이터 유출 같은 실제 행동 유발), ② 거짓 결론을 코퍼스에 쓰지 않고 생성 시점에 합성시키는 은밀한(합성형) 오염, ③ 웹 숨김 콘텐츠를 매개로 한 소셜 웹 인젝션 벤치마크, ④ 탐지가 아닌 정보흐름 제어·격리 기반 방어와 "방어 벤치마크가 지름길 때문에 과대평가되었다"는 메타평가 비판, ⑤ 짧은 QA에서 딥리서치 보고서로 옮겨간 평가(그리고 인용 정확도 지표의 한계 비판), ⑥ 데이터스토어에 대한 형식적 DP 보장이다.

### Cited Findings
- **에이전트까지 이어지는 IPI**: 다중 에이전트 워크플로에서 오염 이메일 1통으로 GPT-4o가 SSH 키를 유출하게 만드는 데 80% 이상 성공했고, 기존 방어는 악성 텍스트의 검색을 막지 못했다. (arXiv 2601.07072. USENIX Security 2026 채택은 ⚠️ 1차 미확인) — [arXiv 2601.07072](https://arxiv.org/abs/2601.07072)
- 같은 USENIX Security 2026에서 에이전트 도구 호출의 인과 귀속으로 IPI를 막는 AttriGuard가 방어 측으로 채택되었다는 이전 판 기재는 ⚠️ **Unverified**다(usenix.org 차단, arXiv ID 미조사). 내용도 RAG보다는 에이전트 범위에 가깝다. — [USENIX PDF](https://www.usenix.org/system/files/usenixsecurity26-he-yu.pdf)
- **은밀한 합성형 오염**: 페이로드를 무해해 보이는 잠복 문단 사슬로 쪼개, 함께 검색될 때 LLM이 다중 홉 추론으로 거짓 결론을 스스로 도출하게 한다. 단일 문서 주입용 방어를 우회한다. (CCS 2026) — [arXiv 2609.16818](https://arxiv.org/abs/2609.16818)
- **웹 숨김 콘텐츠 매개 IPI 벤치마크**: 6,200개 소셜형 페이지, 5종 숨김 매체(본문) (WWW 2026) — [arXiv 2601.10923](https://arxiv.org/abs/2601.10923); [HTML v1](https://arxiv.org/html/2601.10923v1)
- **탐지에서 제어로의 방어 패러다임 전환**: 모순을 탐지하고도 오염 주장대로 행동하는 "monitoring-control gap"을 지적하고 IFC 기반 다중 에이전트 방어를 제안했다. — [arXiv 2605.26754](https://arxiv.org/abs/2605.26754). 후속 연구 "When Detection Does Not Guarantee Resistance: Reasoning and Poisoned Context in RAG"(2608.17153, Mehrdad Ghassabi et al., v1 2026-08-17, Preprint)는 Cordon Rate·Leakage Rate 지표를 정의하고, 200개 SciFact 질문에서 DeepSeek-V4-Flash의 추론 기능을 켜면 Cordon Rate가 0.205→0.075로 떨어지는 반면 ASR은 0.233→0.298로 오른다고 보고한다. — [arXiv](https://arxiv.org/abs/2608.17153) (✅)
- **방어 평가의 메타 비판**: 21개 방어 평가 중 8개가 질문 접두형 오염문서를, 5개가 길이 단서를 사용했고, 12개는 운용 지점 오탐률을 보고하지 않았다. — [arXiv 2610.33849](https://arxiv.org/abs/2610.33849). ASR이 방어로 인한 실패를 숨긴다는 주장과 matched-query 평가 프로토콜도 나왔다. — [arXiv 2610.40742](https://arxiv.org/abs/2610.40742). 조직적 오염에 대한 수집 시점(admission-time) 방어의 근본 한계도 제시되었다("Coverage Is Not Containment", 2608.16044, Prashant Kumar Pathak et al., v1 2026-08-17, Comments "Preprint; under submission"). end-to-end로 대상의 88%에서 심어 둔 주장이 출력되었고, 최강 분류기도 1% FPR에서 공격의 4.2%만 잡았다. — [arXiv](https://arxiv.org/abs/2608.16044) (✅). 계층형 방어 TriShieldRAG(2607.23838)는 v2 Comments에서 v1의 "91%→13%" 헤드라인 결과를 철회했다고 밝혀, 이 분야에서 수치 재현성이 쟁점임을 보여 준다. — [arXiv](https://arxiv.org/abs/2607.23838) (✅)
- **딥리서치 출력 평가**:
  - DeepResearch Bench는 ICLR 2026 포스터로 색인되어 있다(⚠️ 1차 미확인). — [arXiv 2506.11763](https://arxiv.org/abs/2506.11763)
  - DEER는 ICML 2026에 채택되었다(arXiv Comments). 인용된/인용되지 않은 주장 모두에 대한 claim 검증과 증거 품질 정량화를 평가 축으로 추가한다. — [arXiv 2512.17776](https://arxiv.org/abs/2512.17776)
  - 이전 판의 "같은 ICLR 2026에 'Towards Personalized Deep Research: Benchmarks and Evaluations'도 게재"는 ⚠️ **Unverified**다(proceedings.iclr.cc를 이번에 열람하지 않았고 arXiv ID도 미조사). — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/43ec75c1c7de5eb82aa7616a277ba3f4-Abstract-Conference.html)
  - DREAM은 인용 정렬만으로는 사실·추론 결함을 잡을 수 없다는 "Mirage of Synthesis"를 주장하고, 본문 실험에서 DeepResearch Bench의 FACT 인용 정확도가 오염 비율과 무관하게 사실상 일정했음을 보였다. — [arXiv 2602.18940](https://arxiv.org/abs/2602.18940); [HTML v1](https://arxiv.org/html/2602.18940v1)
  - 연구 궤적 전체의 환각을 평가하는 PING 분류 체계와 DeepHalluBench도 나왔다. — [arXiv 2601.22984](https://arxiv.org/abs/2601.22984)
- **LLM-as-judge 평가 표준화**: 고정 예산·클러스터 인식 LLM-as-judge 최소 측정 표준을 제안한 연구가 있다(2605.27789, Camilo Chacón Sartori, José H. García; v1 2026-05-27, Preprint). CS/ML·재료과학 400개 다중 홉 질문으로 스트레스 테스트한 결과, 이항 검정으로는 네 비교가 모두 유의했지만 클러스터 인식 추론에서는 Bonferroni 유의 결과가 하나만 남았다. — [arXiv 2605.27789](https://arxiv.org/abs/2605.27789) (✅). ACL 2026 RAG4Reports 워크숍의 UNH 시스템(Task A 1위)은 인용 충실성이 가장 핵심적인 신호라고 보고한다. — [ACL Anthology 2026.rag4reports-1.9](https://aclanthology.org/2026.rag4reports-1.9/) (✅)
- **형식적 프라이버시 보장**: LSH 기반 DP 데이터스토어로 ε=5에서 평균 정확도 손실 2.6%, MIA 정확도 53.60%를 보고했다(ICPR 2026). — [arXiv 2606.01413](https://arxiv.org/abs/2606.01413)

### Inferences
- 공격 측은 "검색되게 만들기(retrievability)"와 "탐지 회피(stealth)"를 분리했다가 결합하는 방향으로 진화했다(C2의 트리거/공격 조각 분리, C4의 잠복 문단 사슬). 그 결과 문서 단위 필터형 방어의 전제가 약해졌다. 이것이 D1·D2 같은 구조적 격리 방어와 방어 평가 재검토가 2026년에 함께 늘어난 배경으로 보인다.
- RAG 평가의 무게중심은 "정답 일치"에서 "주장 단위 검증 + 인용 + 시간적 유효성"으로 옮겨가고 있으며, 평가자 자체가 도구를 쓰는 에이전트가 되는 추세(DREAM)가 나타났다. 이는 [B] 충실성/귀속 연구와 [F] 평가 연구의 경계를 흐리게 한다.

### Gaps
- 검색 에이전트·딥리서치 에이전트를 직접 겨냥한 오염/인젝션 공격 중 게재처가 확인된 2026 논문은 찾지 못했다. 일반 에이전트 IPI 논문들은 arXiv에서 실재와 Preprint 상태만 확인했다: pikit(2609.36817, Zonghao Ying et al., 13개 공격·16개 매체·9개 방지 전략 툴킷), Scaling Adaptive Prompt Injection Attacks with Agent Swarms(2611.11861, Alexey Kurakin et al.), Blocking at the Boundary(2610.05163, Jingkai Liu et al., Claude Code·Codex 대상 staged prompt injection과 479쌍 벤치마크). — [arXiv 2609.36817](https://arxiv.org/abs/2609.36817), [arXiv 2611.11861](https://arxiv.org/abs/2611.11861), [arXiv 2610.05163](https://arxiv.org/abs/2610.05163)
- NDSS 2026 / IEEE S&P 2026의 RAG 관련 채택 논문은 확인하지 못했다.
- 2026년 "신선도(freshness)" 특화 RAG 벤치마크 중 게재처가 확인된 것은 찾지 못했다(신규·롱테일 지식을 다루는 MINTQA가 가장 가까운 사례).

## Verification log

**방법**: 2026-11-19에 `curl https://arxiv.org/abs/<id>`로 38개 arXiv ID 전부(HTTP 200)를 받고, `citation_title`/`citation_author` 메타, Submission history(v1), Comments, Journal-ref, DOI, 초록을 파싱했다. 초록 밖 수치는 `arxiv.org/html/<id>`(2601.10923v1, 2609.16818v1, 2602.18940v1, 2605.26754v1, 2606.01413v1, 2506.11763v1, 2607.23838v2)에서 확인했다. ACL Anthology 개별 페이지 5건(`2026.findings-acl.338`, `2026.acl-long.282`, `2026.emnlp-main.530`, `2026.acl-long.18`, `2026.rag4reports-1.9`)과 `2026.emnlp-main`·`2026.findings-emnlp` 볼륨 목록을 직접 열었다. export.arxiv.org API는 "Rate exceeded"로 응답해 abs 페이지로 대체했다. OpenReview(포럼·`api.openreview.net/notes?id`는 ChallengeRequiredError, `api2`는 프록시 차단, v1 검색에는 2025–26 기록 없음), usenix.org, iclr.cc, dblp.org는 접근하지 못했다.

**실재 여부**: 목록의 모든 논문(본 목록 14편 + 추가 후보·인용 논문)이 arXiv 또는 ACL Anthology에 실재하며 제목·제1저자·ID가 일치한다. **삭제한 논문은 없다.**

**정정 사항**
1. **E2 DP Datastore (2606.01413)**: 게재처를 Preprint에서 **ICPR 2026**으로 정정했다(arXiv Comments). 확인되지 않은 "목록 공지 2026-06-02"는 삭제했다.
2. **C2 Overcoming the Retrieval Barrier (2601.07072)**: arXiv Comments가 비어 있고 usenix.org에 접근할 수 없어, USENIX Security 2026을 확인 라벨에서 **⚠️ Unverified**로 낮추고 기본 라벨을 Preprint (arXiv)로 바꿨다. 소속(MBZUAI)도 미확인으로 처리했다.
3. **F2 DeepResearch Bench (2506.11763)**: 저자 목록에서 arXiv에 없는 "Licheng Zhang"을 삭제했다(5명). ICLR 2026 Poster는 **⚠️ 1차 미확인**(검색 색인뿐)으로 표기했다. v1 일자 2025-06-13을 확정했다. 요약은 초록 기준으로 다시 쓰고 RACE/FACT 명칭과 50/50 언어 구성은 "(본문)"으로 구분했다. "딥리서치 에이전트가 유효 인용에서 크게 앞섬"은 본문에 맞게 "Grok 제외"로 정정했다.
4. **F3 DREAM (2602.18940)**: 제1저자를 **Elad Ben Avraham**으로 확정하고 저자 11명을 기재했다. 소속(AWS Agentic AI, Georgia Tech 인턴)은 본문으로 확인했다. FACT "≈100%"는 정확한 값을 추출하지 못해 "사실상 일정"으로 완화했다.
5. **C4 InceptionRAG (2609.16818)**: CCS '26은 arXiv Comments로 ✅ 확인했다. 요약을 초록에 맞춰 고쳤다: "두 개의 잠복 문서"를 "잠복 문단의 사슬"로 바꾸고(본문 기본 구성 2개는 별도 표기), ZOSO와 3개 데이터셋 × 3개 LLM, 80% 초과를 반영했다. HODOR의 다수결 방식과 다중 홉 성능 저하는 본문으로 확인했다. 인용구는 원문 문구로 교체했다.
6. **C3 Hidden-in-Plain-Text (2601.10923)**: WWW 2026은 arXiv Comments와 DOI로 ✅ 확인했다. 6,200 페이지와 5종 매체는 초록이 아니라 본문 §3.1 내용이므로 그렇게 표기하고 매체별 수치를 추가했다. 확인되지 않은 "정상 명령문 hard negative"는 삭제했다.
7. **D1 Cordon-MAS (2605.26754)**: arXiv 저자 7명이 이전 기재와 일치함을 확인해 우려를 해소했다. v1 2026-05-26을 확정하고 Cordon 원칙과 Gate 구성요소를 추가했다. TriShieldRAG의 "3–4회 순차 LLM 호출"은 v2 본문에서 확인했다.
8. **D2 Ranking Is Not Deciding (2610.33849)**: v1 2026-10-24를 확정했다. "5개는 길이 단서", RAGuard 미탐률 0.191→0.763, TrustRAG 0.000→0.827, "방어를 주장하지 않음"을 추가했다.
9. **A1 MADAM-RAG (2504.13079)**: 초록에 없는 "Astute RAG 대비"를 삭제했다. FaithEval 최대 15.80%와 RAMDocs 기준선 EM 32.60을 추가했다. v1 2025-04-17을 확정했다.
10. **B2 CiteGuard (2510.17853)**: 초록에 없는 "검색 지식 가정" 설명과 "Semantic Scholar API"를 삭제하고, "인간 저자가 달았을 인용과의 일치"라는 정의로 바꿨다. 확인하지 않은 "arXiv v1은 65.4% vs 69.7%"도 삭제했다. ACL 2026 Main은 arXiv Comments와 Anthology로 ✅ 확인했다.
11. **B1 FRANQ (2505.21072)**: 최신 버전을 v3에서 v5(2026-04-29)로 정정했다. Findings of ACL 2026과 pp. 6814–6836은 Anthology로 ✅ 확인했다.
12. **F1 Great Nugget Recall (2504.15068)**: 초록에 없는 "nugget recall만 평가" 단서를 삭제하고, 초록의 단서인 "주제 수준 일치는 추가 연구 필요"로 바꿨다. v1 2025-04-21을 확정했다.
13. **E1 Riddle Me This (2502.00306)**: 탐지 빈도 약 76배 차이를 추가했다. CCS 2025는 arXiv Comments로 ✅ 확인했다.
14. **C1 PoisonedRAG (2402.07867)**: v1 2024-02-12를 확정했다. USENIX Security 2025는 arXiv Comments로 ✅ 확인했고, 쪽수(pp. 3827–3844)는 ⚠️ 미확인이다.
15. **추가 후보**:
    - FIDES 설명을 전면 정정했다(학습 불필요 토큰 단위 대조 디코더).
    - EARAL의 arXiv가 2508.08942(LoDIT)임을 확정했다(arXiv DOI 필드와 저자 일치).
    - SDAG를 **EMNLP 2026 Main**(`2026.emnlp-main.1399`)으로, DEER를 **ICML 2026**(arXiv Comments)으로 게재처 추가했다.
    - Survey of Secure RAG를 **Findings of EMNLP 2026**(`2026.findings-emnlp.241`)으로 표기했다.
    - ConflictRAG는 "submitted to IEEE SMC 2026"이므로 Preprint로 두고, 근거 없는 "수치 불일치" 언급을 삭제했다.
    - Confundo와 DEER의 설명을 초록에 맞게 고쳤다.
    - ProbeRAG(2510.12460)를 ACL 2026 Findings로 식별했다.
16. **⚠️ Unverified로 남은 항목**:
    - 2601.07072의 USENIX Security 2026
    - DeepResearch Bench의 ICLR 2026 Poster
    - AttriGuard의 USENIX Security 2026
    - "Towards Personalized Deep Research"의 ICLR 2026
    - PoisonedRAG 쪽수
    - WWW 2026 트랙
    - Xu et al. 2024 지식 충돌 분류의 원문 문장

    허용 목록에 usenix.org, iclr.cc를 추가하거나 OpenReview 챌린지를 우회할 수 있을 때 재확인이 필요하다.
