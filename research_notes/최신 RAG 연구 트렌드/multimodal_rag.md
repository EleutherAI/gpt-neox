# Multimodal RAG: 최신 논문 큐레이션 (기준일 2026-11-19)

> **검증 방법 및 한계 (보고서 작성자 필독). 2026-11-19 2차 검증 반영**
> - 1차 조사에서는 arxiv.org, openreview.net, aclanthology.org, CVF에 접속하지 못해 검색 스니펫과 저자 GitHub README에 의존했다.
> - 2차 검증(이번 판)에서는 노트에 나온 **모든 arXiv ID의 arxiv.org/abs 페이지를 직접 받아** 제목, 저자, 제출 이력(v1 날짜), Comments, 초록을 확인했다. 필요한 경우 arXiv PDF 첫 페이지 헤더도 확인했다. ACL Anthology는 **개별 논문 페이지와 2025–2026 ACL/EMNLP Main·Findings 볼륨 목록 전체**를 받아 제목으로 대조했다.
> - 2차 검증 때도 접근하지 못한 곳이 있다.
>   - export.arxiv.org API는 "Rate exceeded"를 반환해서 abs 페이지로 대체했다.
>   - OpenReview: api.openreview.net은 `ChallengeRequiredError`(403 봇 챌린지)를 반환했다. api2.openreview.net은 프록시가 차단했고, openreview.net 포럼은 챌린지로 리다이렉트되었다.
>   - iclr.cc, neurips.cc, icml.cc, cvpr.thecvf.com, openaccess.thecvf.com, proceedings.mlr.press, proceedings.neurips.cc는 모두 프록시가 CONNECT 403으로 차단했다.
>   - 그래서 ICLR·NeurIPS·ICML·CVPR 채택 여부는 **arXiv의 Comments 필드나 PDF 헤더(저자가 기재한 1차 페이지 정보)**로 판정했다. 이마저 없으면 **Unverified**로 표시했다.
> - 각 논문 아래 "검증:" 줄에 확인 결과와 venue 출처를 적었다. ✅는 1차 페이지에서 확인했다는 뜻이고, ⚠️는 일부만 확인되었거나 저자 공지에만 근거한다는 뜻이다. 정정 내역은 맨 끝 "Verification log"에 모았다.
> - 날짜는 arXiv 제출 이력의 v1 날짜(UTC)다.

## Q1. 가장 영향력 있는 논문과 최신(2025–2026) Multimodal RAG 논문은 무엇이며, 각각 어느 학회에 채택되었나?

### Takeaway
이 분야의 출발점은 페이지를 이미지로 검색하는 문서 RAG다. ColPali와 VisRAG는 ICLR 2025, VDocRAG는 CVPR 2025에 채택되었다. 2025년 하반기부터 2026년 사이에는 세 갈래로 넓어졌다.
- (a) 다중 이미지 증거를 고르고 추론하는 연구: ACL 2026 Main, EMNLP 2026 Main. EVisRAG의 ACM MM 2026은 저자 공지만 있다.
- (b) 비디오·오디오 RAG: NeurIPS 2025, ACL 2025 Main, ACL 2026 Findings.
- (c) RL로 학습한 멀티모달 검색·딥리서치 에이전트: MMSearch-R1(ACL 2026 Main), Video-DeepResearch(EMNLP 2026 Findings), ODE(NeurIPS 2026, arXiv Comments 기준). Vision-DeepResearch의 ICML 2026은 저자 공지만 있고 1차 proceedings로는 확인하지 못했다.

2026년 최신 논문 중 상당수는 아직 arXiv 프리프린트다.

### Cited Findings

#### 핵심 큐레이션 목록 (12편)

**[F1] ColPali: Efficient Document Retrieval with Vision Language Models** (기초 논문)
- 저자: Manuel Faysse et al. (7명)
- arXiv: [2407.01449](https://arxiv.org/abs/2407.01449). 최초 공개(v1): 2024-06-27. 최신 v6: 2025-02-28.
- 학회: **ICLR 2025**. 근거는 arXiv Comments "Published as a conference paper at ICLR 2025"다 ([arXiv](https://arxiv.org/abs/2407.01449)).
- 요약:
  1. 시각적으로 풍부한 문서 검색을 평가하는 페이지 단위 벤치마크 ViDoRe를 제안했다. 여러 도메인·언어·실사용 설정을 포괄한다. 또 문서 페이지 이미지에서 바로 multi-vector 임베딩을 만드는 VLM 검색기 ColPali를 공개했다.
  2. late-interaction 매칭과 결합한 ColPali는 텍스트 추출에 의존하는 기존 문서 검색 파이프라인보다 성능이 크게 높다. 동시에 훨씬 단순하고 빠르며 end-to-end로 학습할 수 있다. 모델·데이터·코드·벤치마크는 hf.co/vidore에 공개되어 있다.
- 인용: "ColPali largely outperforms modern document retrieval pipelines while being drastically simpler, faster and end-to-end trainable" ([arXiv](https://arxiv.org/abs/2407.01449))
- 검증: ✅ arXiv abs 확인. 제목, 첫 저자, 초록이 일치한다. v1 날짜는 "2024-07(ID 기준)"에서 2024-06-27로 정정했다. / venue 출처: arXiv Comments. iclr.cc와 OpenReview는 접속이 차단되어 다시 확인하지 못했다.

**[F2] VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents** (기초 논문)
- 저자: Shi Yu et al. (11명)
- arXiv: [2410.10594](https://arxiv.org/abs/2410.10594). 최초 공개(v1): 2024-10-14. v2: 2025-03-02.
- 학회: **ICLR 2025**. 근거는 arXiv PDF v2 첫 페이지 헤더 "Published as a conference paper at ICLR 2025"다 ([arXiv PDF v2](https://arxiv.org/pdf/2410.10594v2)). arXiv Comments 필드는 비어 있다. 1차 노트의 "Poster" 구분과 OpenReview ID zG459X3Xge는 OpenReview가 차단되어 다시 확인하지 못했다.
- 요약:
  1. 문서를 먼저 텍스트로 파싱하지 않는다. VLM으로 페이지를 이미지 그대로 임베딩·검색하고, 검색된 이미지로 VLM이 답을 생성하는 RAG 파이프라인이다. 이렇게 해서 파싱 과정의 정보 손실을 없앴다. 검색기 학습에는 공개 데이터와 합성 데이터를 함께 썼고, 여러 생성 방식을 탐색했다.
  2. 검색과 생성 양쪽에서 텍스트 기반 RAG보다 성능이 높았다. 현재 v2 초록은 end-to-end 성능이 **20–40%** 향상되었다고 적고 있고, v1 초록은 25–39%였다. 학습 데이터를 효율적으로 쓰고 일반화 능력도 강했다.
- 인용: "the document is directly embedded using a VLM as an image and then retrieved to enhance the generation of a VLM" ([arXiv](https://arxiv.org/abs/2410.10594))
- 검증: ✅ arXiv abs 확인. 제목, 첫 저자, v1 날짜가 일치한다. 수치는 25–39%(v1)와 20–40%(v2)로 버전마다 달랐음을 확인했다. / venue 출처: arXiv PDF v2 헤더.

**[F3] VDocRAG: Retrieval-Augmented Generation over Visually-Rich Documents** (기초 논문, 2025 상반기)
- 저자: Ryota Tanaka et al. (6명)
- arXiv: [2504.09795](https://arxiv.org/abs/2504.09795). 최초 공개(v1): 2025-04-14.
- 학회: **CVPR 2025**. 근거는 arXiv Comments "Accepted by CVPR 2025; project page: …"다 ([arXiv](https://arxiv.org/abs/2504.09795)). 1차 노트의 CVF Open Access 페이지(pp. 24827–24837)는 프록시 차단으로 다시 확인하지 못했다.
- 요약:
  1. 차트·표 등이 섞이고 PDF·PPTX처럼 형식이 다양한 문서 코퍼스를 하나의 이미지 형식으로 다루는 RAG 프레임워크다. 시각 정보를 dense 토큰 표현으로 압축하면서 문서 텍스트와 정렬하는 self-supervised 사전학습 과제를 만들어, LVLM을 검색기로 적응시켰다.
  2. 개방 도메인 문서 VQA 데이터셋을 처음으로 통합한 OpenDocVQA를 공개했다. 실험에서 VDocRAG는 텍스트 기반 RAG보다 성능이 크게 높았고 일반화 능력도 강했다.
- 인용: "the first unified collection of open-domain document visual question answering datasets" ([arXiv](https://arxiv.org/abs/2504.09795))
- 검증: ✅ arXiv abs 확인. 제목, 첫 저자, 초록이 일치한다. v1 날짜는 "2025-04(ID 기준)"에서 2025-04-14로 구체화했다. / venue 출처: arXiv Comments. CVF는 접속이 차단되었다.

**[4] VisRAG 2.0 / EVisRAG: Mitigating Visual Hallucinations via Evidence-Guided Multi-Image Reasoning in Visual RAG**
- 제목 변경 이력:
  - v1: "VisRAG 2.0: Evidence-Guided Multi-Image Reasoning in Visual Retrieval-Augmented Generation" ([arXiv v1](https://arxiv.org/abs/2510.09733v1))
  - 현재 v2: "VisRAG2.0: Mitigating Visual Hallucinations via Evidence-Guided Multi-Image Reasoning in Visual Retrieval-Augmented Generation" ([arXiv](https://arxiv.org/abs/2510.09733))
- 저자: **Yubo Sun** et al. (8명, Shi Yu는 4번째 저자)
- arXiv: [2510.09733](https://arxiv.org/abs/2510.09733). 최초 공개(v1): **2025-10-10**. v2: 2026-07-28.
- 학회: **Unverified**. 저자 repo에만 "20260709: Our work EVisRAG is accepted by ACM MM 2026"이라는 공지가 있다 ([GitHub](https://github.com/OpenBMB/VisRAG)). arXiv Comments는 비어 있고, v2 PDF 첫 페이지에도 ACM MM 표기가 없다. ACM DL은 확인하지 못했다. 1차 페이지 기준 상태는 Preprint (arXiv)다.
- 요약:
  1. 다중 이미지 Visual RAG에서 VLM은 시각적 환각을 자주 일으키고 질문 관련 증거를 정확히 찾지 못한다. 기존 방법에는 이미지 간 증거를 명시적으로 모으는 과정이 없고, 지각과 추론을 함께 최적화할 때 credit assignment도 약하다. EVisRAG는 검색된 이미지를 하나씩 관찰해 질문 관련 증거를 기록하고, 모은 증거로 추론·답변한다. 학습에는 단계별 토큰 구간에 보상을 맞추는 RS-GRPO를 썼다.
  2. 여러 VQA 벤치마크에서 백본 VLM보다 **평균 약 19%**(v2 초록) 높았고, 시각적 환각을 크게 줄였다. v1 초록은 평균 27% 향상으로 적고 있었다.
- 인용: "records question-relevant visual evidence from each image, and then performs reasoning and answer generation based on the aggregated evidence" ([arXiv](https://arxiv.org/abs/2510.09733))
- 검증: ⚠️ arXiv abs는 확인했다(제목, 저자, 날짜, 초록). 첫 저자는 미확인에서 Yubo Sun으로, v1 날짜는 2025-10-14에서 2025-10-10으로 정정했다. / venue 출처: 저자 repo 공지뿐이라 Unverified다.

**[5] Utility-Oriented Visual Evidence Selection for Multimodal Retrieval-Augmented Generation**
- 저자: Weiqing Luo et al. (6명)
- arXiv: [2605.13277](https://arxiv.org/abs/2605.13277). 최초 공개(v1): 2026-05-13.
- 학회: **ACL 2026 Main** (Volume 1: Long Papers, pp. 35091–35124). 근거: [ACL Anthology 2026.acl-long.1620](https://aclanthology.org/2026.acl-long.1620/), arXiv Comments "Accepted to ACL 2026"
- 요약:
  1. 기존 시각 증거 선택은 의미적 관련성이나 표면적 유사도에 기대는데, 이는 실제 추론에 주는 효용과 어긋난다. 이 논문은 증거 효용을 모델 출력 분포에 대한 정보 이득으로 정의했다. 잠재 변수인 "helpfulness"로 순위를 매기는 것이 약한 가정 아래에서 답 공간 효용과 같다는 것도 이론으로 보였다.
  2. 가벼운 멀티모달 모델로 효용을 추정하는 training-free, surrogate 가속 프레임워크를 제안했다. MRAG-Bench와 Visual-RAG에서 여러 모델 계열에 걸쳐 최신 RAG 기준선보다 일관되게 높았고, 계산 비용은 크게 줄었다.
- 인용: "existing methods typically rely on semantic relevance or surface-level similarity" ([arXiv](https://arxiv.org/abs/2605.13277), [ACL Anthology](https://aclanthology.org/2026.acl-long.1620/))
- 검증: ✅ arXiv abs와 ACL Anthology 확인. v1 날짜는 2026-05-13으로 구체화했다. 1차 요약의 "Top-K만 큰 모델에 넘김"은 초록에 명시되지 않아 뺐다. / venue 출처: ACL Anthology (acl-long).

**[6] Doc-REFRAG: Rethinking Multimodal Document Retrieval-Augmented Generation**
- 저자: Ruofan Hu et al. (8명, 마지막 저자 Zhou Zhao)
- arXiv: [2608.30163](https://arxiv.org/abs/2608.30163). 최초 공개(v1): 2026-08-31.
- 학회: **EMNLP 2026 Main** (pp. 8624–8644). 근거: [ACL Anthology 2026.emnlp-main.377](https://aclanthology.org/2026.emnlp-main.377/), arXiv Comments "Accepted by EMNLP 2026 Main"
- 요약:
  1. 기존 멀티모달 RAG는 단일 이미지나 닫힌 문서 설정에 맞춰져 있어 실제 다중 이미지 상황에서 정확도가 낮다. 또 검색 이미지가 많아지면 무관한 시각 토큰 때문에 계산 비용이 커진다. 이를 풀기 위해 질문마다 평균 37.4장의 검색 이미지가 붙은 343K QA 데이터셋 DocLongRAG를 만들었다.
  2. Doc-REFRAG는 시각 토큰을 거친 청크로 압축해 두고, 가벼운 RL 기반 선택기가 질문 관련 청크만 다시 펼친다. 6개 벤치마크에서 11개 강력한 기준선을 넘는 SOTA 정확도를 기록했고, 추론 지연은 훨씬 낮았다.
- 인용: "Real-world knowledge resides in multimodal documents, necessitating retrieval-augmented generation (RAG) for accurate question answering." ([arXiv](https://arxiv.org/abs/2608.30163))
- 검증: ✅ arXiv abs와 ACL Anthology 확인. v1 날짜는 2026-08-31로 구체화했다. 1차 요약의 "디코딩 중 원해상도 토큰으로 확장"은 초록 표현("selectively expands question-relevant ones")에 맞게 고쳤다. / venue 출처: ACL Anthology (emnlp-main)와 arXiv Comments.

**[7] Vgent: Graph-based Retrieval-Reasoning-Augmented Generation For Long Video Understanding**
- 저자: Xiaoqian Shen et al. (4명, 마지막 저자 Mohamed Elhoseiny)
- arXiv: [2510.14032](https://arxiv.org/abs/2510.14032). 최초 공개(v1): 2025-10-15.
- 학회: **NeurIPS 2025 (Spotlight)**. 근거는 arXiv Comments "NeurIPS 2025 (Spotlight)"다 ([arXiv](https://arxiv.org/abs/2510.14032)). neurips.cc와 OpenReview는 차단되어 다시 확인하지 못했다.
- 요약:
  1. 긴 비디오에 RAG를 적용하면 시간적 의존성이 끊기고 무관한 정보가 섞인다. Vgent는 클립 사이의 의미 관계를 보존하는 그래프로 비디오를 표현해 검색 효과를 높인다. 또 구조화된 검증으로 검색 잡음을 줄이고 클립 간 정보를 명시적으로 모으는 중간 추론 단계를 추가했다.
  2. 여러 오픈소스 LVLM과 장편 비디오 벤치마크 3종으로 평가했다. MLVU에서는 기본 모델 대비 3.0–5.4% 향상되었고, 최신 비디오 RAG 방법보다 8.6% 높았다.
- 인용: "a novel graph-based retrieval-reasoning-augmented generation framework to enhance LVLMs for long video understanding" ([arXiv](https://arxiv.org/abs/2510.14032))
- 검증: ✅ arXiv abs 확인. 1차 요약의 "학습 불필요 4단계", "2B–7B", "LongVideoBench·Video-MME"는 초록에 없어 뺐다. 초록에 이름이 나온 벤치마크는 MLVU뿐이다. / venue 출처: arXiv Comments.

**[8] WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models**
- 저자: **Yifu Chen** et al. (8명, 마지막 저자 Zhou Zhao)
- arXiv: [2502.14727](https://arxiv.org/abs/2502.14727). 최초 공개(v1): 2025-02-20.
- 학회: **ACL 2025 Main** (Volume 1: Long Papers, pp. 12505–12523). 근거: [ACL Anthology 2025.acl-long.613](https://aclanthology.org/2025.acl-long.613/)
- 요약:
  1. 기존 RAG는 음성을 ASR로 처리하기 때문에 오디오 정보를 잃고, 전사 오류가 생기며, 계산 비용이 늘어난다. WavRAG는 ASR을 거치지 않고 원시 오디오를 바로 임베딩·검색한다. WavRetriever로 텍스트·오디오 혼합 지식베이스에서 검색하고, chain-of-thought를 결합해 음성 대화 모델의 in-context 능력을 높였다.
  2. 최신 ASR-Text RAG 파이프라인과 비슷한 검색 성능을 내면서 속도는 10배 빠르다. 텍스트·오디오 혼합 검색으로 RAG를 오디오 모달리티까지 넓혔다.
- 인용: "the first retrieval augmented generation framework with native, end-to-end audio support" ([arXiv](https://arxiv.org/abs/2502.14727))
- 검증: ✅ arXiv abs와 ACL Anthology 확인. 첫 저자는 미확인에서 Yifu Chen으로, v1은 2025-02-20으로 구체화했다. 초록 수치는 10x이고, 1차 노트의 "5–14배 불일치" 언급은 초록 근거가 없어 지웠다. / venue 출처: ACL Anthology (acl-long).

**[9] PlanRAG-Audio: Planning and Retrieval Augmented Generation for Long-form Audio Understanding**
- 저자: Masao Someki et al. (10명, 마지막 저자 Shinji Watanabe)
- arXiv: [2605.20414](https://arxiv.org/abs/2605.20414). 최초 공개(v1): 2026-05-19. v2: 2026-05-25.
- 학회: **ACL 2026 Findings** (pp. 26167–26183). 근거: [ACL Anthology 2026.findings-acl.1304](https://aclanthology.org/2026.findings-acl.1304/), arXiv Comments "Accepted to Findings of ACL 2026"
- 요약:
  1. 장시간 오디오는 시퀀스가 매우 길고, 발화 내용·화자·감정·음향 이벤트 단서가 시간축에 흩어져 있다. PlanRAG-Audio는 녹음 전체를 LALM에 넣는 대신, 질의에 필요한 모달리티와 시간 구간을 먼저 계획하고 구조화된 텍스트·오디오 DB에서 그 정보만 검색한다.
  2. LLM에 넘기는 입력 길이를 크게 줄이면서 추론 정확도를 높였다. 추론 비용을 원시 오디오 길이와 분리했기 때문에 오디오가 길어져도 성능이 안정적이다.
- 인용: "a planning-based retrieval-augmented generation framework for scalable long-form audio understanding" ([arXiv](https://arxiv.org/abs/2605.20414))
- 검증: ✅ arXiv abs와 ACL Anthology 확인. 인용구는 초록 원문과 일치한다. 1차 요약의 "작업별 프롬프트·수작업 SQL 없이 zero-shot"은 초록에 없어 뺐다. / venue 출처: ACL Anthology (findings-acl)와 arXiv Comments.

**[10] MMSearch-R1: Incentivizing LMMs to Search**
- 저자: Jinming Wu et al. (8명, 마지막 저자 Ziwei Liu)
- arXiv: [2506.20670](https://arxiv.org/abs/2506.20670). 최초 공개(v1): **2025-06-25**.
- 학회: **ACL 2026 Main** (Volume 1: Long Papers, pp. 2456–2487). 근거: [ACL Anthology 2026.acl-long.114](https://aclanthology.org/2026.acl-long.114/), [저자 repo "[26.04.07] … accepted to ACL-2026 Main conference"](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)
- 요약:
  1. 고정된 RAG 파이프라인과 프롬프트 기반 검색 에이전트는 검색을 비효율적으로 하거나 과도하게 한다. MMSearch-R1은 실제 인터넷 환경에서 LMM이 필요할 때 여러 차례 검색하도록 만든 최초의 end-to-end RL 프레임워크다. 이미지·텍스트 검색 도구를 붙이고, 결과 기반 보상에 검색 페널티를 더해 언제·어떻게 검색할지 학습시켰다. 반자동으로 모은 멀티모달 검색 VQA 데이터에서 검색이 필요한 예와 필요 없는 예를 균형 있게 고른 부분집합이 효율적인 검색 행동을 만드는 데 핵심이었다.
  2. 지식 집약형·정보 탐색형 VQA에서 같은 크기의 RAG 기준선을 넘었다. 더 큰 RAG 기반 모델과는 비슷한 성능을 내면서 검색 호출을 30% 이상 줄였다.
- 인용: "perform on-demand, multi-turn search in real-world Internet environments" ([arXiv](https://arxiv.org/abs/2506.20670))
- 검증: ✅ arXiv abs와 ACL Anthology 확인. v1은 2025-06-26에서 2025-06-25로 정정했다. "검색 30% 감소"는 같은 크기 기준선이 아니라 더 큰 RAG 모델과 비교한 수치라서 정정했다. "7B 모델·FVQA 공개"는 초록에 없어 뺐다. / venue 출처: ACL Anthology (acl-long).

**[11] Vision-DeepResearch: Incentivizing DeepResearch Capability in Multimodal Large Language Models**
- 저자: Wenxuan Huang et al. (17명, 마지막 저자 Wanli Ouyang)
- arXiv: [2601.22060](https://arxiv.org/abs/2601.22060). 최초 공개(v1): 2026-01-29. 최신 v3: 2026-03-23.
- 학회: **Unverified**. 저자 repo에만 "[2026/05/01] Vision-DeepResearch has been accepted by ICML 2026"이라는 공지가 있다 ([GitHub](https://github.com/Osilly/Vision-DeepResearch)). arXiv Comments는 비어 있고, v3 PDF 헤더에도 학회 표기가 없다. icml.cc, OpenReview, PMLR은 차단되었다. 1차 페이지 기준 상태는 Preprint (arXiv)다.
- 요약:
  1. 기존 "추론 후 도구 호출" 방식의 멀티모달 검색은 단일 전체 이미지나 개체 수준 이미지 질의와 몇 개의 텍스트 질의로 충분하다고 가정한다. 이 가정은 시각 잡음이 많은 현실에서는 맞지 않고, 추론 깊이와 검색 폭도 제한된다. 이 논문은 다회차·다개체·다중 스케일의 시각·텍스트 검색을 수행하는 멀티모달 딥리서치 패러다임을 제안했다.
  2. 추론 단계 수십 회와 검색엔진 상호작용 수백 회를 지원한다. cold-start 지도학습과 RL로 딥리서치 능력을 MLLM 안에 내재화했다. 기존 멀티모달 딥리서치 MLLM과, GPT-5·Gemini-2.5-pro·Claude-4-Sonnet 기반 워크플로를 크게 앞섰다.
- 인용: "performs multi-turn, multi-entity and multi-scale visual and textual search to robustly hit real-world search engines under heavy noise" ([arXiv](https://arxiv.org/abs/2601.22060))
- 검증: ⚠️ arXiv abs는 확인했다. v1은 2026-01-29로 구체화했다. 1차 요약의 "8B·30B-A3B", "사실 VQA 벤치마크 6종 SOTA", "최초의 장기 MLLM"은 초록에 없어 뺐다. / venue 출처: 저자 repo 공지뿐이라 Unverified다.

**[12] Scaling Beyond Context: A Survey of Multimodal Retrieval-Augmented Generation for Document Understanding** (서베이)
- 저자: Sensen Gao et al. (10명, 마지막 저자 Mingming Gong)
- arXiv: [2510.15253](https://arxiv.org/abs/2510.15253). 최초 공개(v1): 2025-10-17. v2: 2026-01-09. v3: 2026-04-20.
- 학회: **ACL 2026 Main** (Volume 1: Long Papers, pp. 4458–4489). 근거: [ACL Anthology 2026.acl-long.204](https://aclanthology.org/2026.acl-long.204/), arXiv Comments "Accepted by ACL2026 Main Conference"
- 요약:
  1. OCR 파이프라인은 구조 정보를 잃고, 네이티브 MLLM은 컨텍스트 모델링에 어려움을 겪는다. 이 서베이는 두 한계를 넘는 대안으로 문서 이해용 멀티모달 RAG를 체계적으로 정리했다. 분류 축은 도메인, 검색 모달리티, 검색 단위(granularity)이고, 그래프 구조와 에이전트 프레임워크 관련 진전도 함께 다룬다.
  2. 주요 데이터셋·벤치마크·응용·산업 배치를 요약했다. 효율성, 세밀한 표현, 강건성을 남은 과제로 꼽고 문서 AI의 로드맵을 제시한다.
- 인용: "current multimodal RAG benchmarks require 20–200M visual tokens, far exceeding the typical 128K–1M context limits of existing MLLMs" ([arXiv HTML v2 본문](https://arxiv.org/html/2510.15253v2)). 초록이 아니라 본문 서론에 있는 문장이다.
- 검증: ✅ arXiv abs와 ACL Anthology 확인. 20–200M 인용구는 HTML v2 본문에서 원문을 확인했다. 1차 요약의 "멀티모달 RAG와 문서 이해를 연결한 첫 서베이"라는 주장은 초록에 없고, HTML v2에서도 해당 표현을 찾지 못해 뺐다. / venue 출처: ACL Anthology (acl-long)와 arXiv Comments.

#### 보조 후보 (채택 상태 포함, 보고서에 선택적으로 사용)
- **Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation** (arXiv [2502.08826](https://arxiv.org/abs/2502.08826), Mohammad Mahdi Abootorabi et al., v1 2025-02-12)
  - 학회: **ACL 2025 Findings** ([ACL Anthology 2025.findings-acl.861](https://aclanthology.org/2025.findings-acl.861/))
  - 텍스트·이미지·오디오·비디오를 아우르는 멀티모달 RAG 서베이다. 데이터셋·벤치마크·지표, 검색·융합·증강·생성 기법, 학습 전략, 강건성, 에이전트 기반 접근을 다룬다.
  - 검증: ✅ arXiv abs와 Anthology 확인 / venue 출처: ACL Anthology. 1차 노트는 저자 repo 공지에만 근거했었다.
- **MRAG-Bench: Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models** (arXiv [2410.08182](https://arxiv.org/abs/2410.08182), Wenbo Hu et al., v1 2024-10-10)
  - 학회: **ICLR 2025** (arXiv Comments "ICLR 2025", 저자 repo도 같은 내용)
  - 9개 시나리오에 걸쳐 이미지 16,130장과 사람이 주석한 객관식 문항 1,353개로 구성된다. 평가한 모든 LVLM이 텍스트보다 이미지로 보강했을 때 더 크게 향상되었다. GPT-4o는 정답 정보를 줘도 5.82%만 올랐고, 사람은 33.16% 올랐다.
  - 검증: ✅ arXiv abs 확인 / venue 출처: arXiv Comments. OpenReview는 차단되었다.
- **UniversalRAG: Retrieval-Augmented Generation over Corpora of Diverse Modalities and Granularities** (arXiv [2504.20734](https://arxiv.org/abs/2504.20734), Woongyeong Yeo et al., v1 2025-04-29)
  - 학회: **ACL 2026 Main** ([ACL Anthology 2026.acl-long.177](https://aclanthology.org/2026.acl-long.177/), arXiv Comments "ACL 2026")
  - 모든 모달리티를 하나의 표현 공간에 몰아넣으면 질의와 같은 모달리티만 선호하는 modality gap이 생긴다. 이를 피하려고 모달리티별 코퍼스로 질의를 라우팅하고, 각 모달리티를 여러 단위(granularity)로 나눴다. 10개 벤치마크에서 검증했다.
  - 검증: ✅ arXiv abs와 Anthology 확인 / venue 출처: ACL Anthology. 1차 노트의 "Unverified"를 정정했다. OpenReview forum mXtV92SPxW는 차단되어 확인하지 못했다.
- **VRAG-RL** (arXiv [2505.22019](https://arxiv.org/abs/2505.22019), Qiuchen Wang et al., v1 2025-05-28)
  - 학회: **Preprint (arXiv)**. arXiv Comments가 없고, 2025–2026 ACL/EMNLP 볼륨에도 없으며, 저자 README에도 채택 공지가 없다. OpenReview는 차단되어 NeurIPS·ICLR 여부는 **Unverified**다.
  - 시각적으로 풍부한 정보를 위한 RL 프레임워크다. crop·scale 행동 공간으로 coarse-to-fine 지각을 하면서 검색엔진과 상호작용한다.
  - 검증: ✅ arXiv abs 확인 / venue: 확인 불가
- **VimRAG: Navigating Massive Visual Context in Retrieval-Augmented Generation via Multimodal Memory Graph** (arXiv [2602.12735](https://arxiv.org/abs/2602.12735), Qiuchen Wang et al., v1 2026-02-13)
  - 학회: **Preprint (arXiv)**. Comments, Anthology, README 모두에 채택 정보가 없다.
  - 텍스트·이미지·비디오를 대상으로 추론 과정을 동적 DAG 메모리로 구조화한다. Graph-Modulated Visual Memory Encoding으로 중요한 증거에만 고해상도 토큰을 배분하고, Graph-Guided Policy Optimization(GGPO)으로 학습한다.
  - 검증: ✅ arXiv abs 확인 / venue: 확인 불가
- **ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents** (arXiv [2502.18017](https://arxiv.org/abs/2502.18017), Qiuchen Wang et al., v1 2025-02-25)
  - 학회: **EMNLP 2025 Main** ([ACL Anthology 2025.emnlp-main.464](https://aclanthology.org/2025.emnlp-main.464/))
  - 다중 에이전트 RAG다. GMM 기반 하이브리드(텍스트+시각) 검색과, 탐색·요약·성찰로 이루어진 반복 에이전트 워크플로를 쓴다. ViDoSeek 벤치마크를 제안했고, 그 위에서 기존 방법보다 10% 이상 높았다.
  - 검증: ✅ arXiv abs와 Anthology 확인 / venue 출처: ACL Anthology. 1차 노트의 "Unverified"를 정정했다. "actor-critic"이라는 표현은 초록에 없어 고쳤다.
- **MR-RAG: Multimodal Relevance-Aware Retrieval-Augmented Generation for Medical Visual Question Answering**
  - 학회: **CVPR 2026** (⚠️ 직접 확인 못 함). 검색엔진이 색인한 [CVPR 2026 포스터 페이지](https://cvpr.thecvf.com/virtual/2026/poster/36898)의 제목·요약에만 근거한다. cvpr.thecvf.com과 openaccess.thecvf.com은 프록시 차단으로 직접 받지 못했고, arXiv 제목 검색 결과는 0건이었다.
  - 의료 VQA용 2단계 RAG다. Multimodal Cooperative Retrieval과 Importance-Aware Information Flow Augmentation으로 구성되고, 세 의료 데이터셋에서 최대 6.4% 향상되었다(검색 스니펫 기준).
  - 검증: ⚠️ arXiv에 없다. CVF 공식 페이지는 색인으로만 존재를 확인했다. / venue 출처: 검색 색인된 CVF 포스터 페이지.
- **MDocRAG-RL: Empowering Multi-Modal Document RAG via Complex Visual Reasoning with Reinforcement Learning** (Zhongyu Wang, Anthology 메타데이터상 단독 저자)
  - 학회: **ACL 2026 Findings** ([ACL Anthology 2026.findings-acl.420](https://aclanthology.org/2026.findings-acl.420/), pp. 8641–8651)
  - 사전학습·파인튜닝 과제로 MLLM이 시각 문서 표현을 압축하고 텍스트·시각 임베딩을 정렬하도록 해 검색 효율을 높였다. 생성기에는 coarse-to-fine 시각 지각 행동 공간을 주고 RL로 복잡한 시각 추론 능력을 강화했다(Anthology 초록).
  - 검증: ✅ ACL Anthology 페이지와 초록 확인. arXiv 링크는 없다. / venue 출처: ACL Anthology. 1차 노트의 "Unverified"를 정정했다.
- **Rethinking RAG in Long Videos: What to Retrieve and How to Use It?** (arXiv [2606.13141](https://arxiv.org/abs/2606.13141), Yuho Lee et al., v1 2026-06-11)
  - 학회: **Preprint (arXiv)**
  - V-RAGBench를 제안했다. 시간 단위 영상에서 각 답이 고유한 증거 청크 하나에 의존하도록 만든 〈query, evidence chunk, answer〉 triplet 벤치마크로, 검색과 생성을 따로 평가할 수 있다. 함께 제안한 CARVE는 training-free 방식으로, 여러 모달리티·단위 설정의 검색기를 병렬로 돌리고 청크마다 적응적으로 재순위화한다. 8개 VideoRAG 기준선보다 두 단계 모두에서 높았다.
  - 검증: ✅ arXiv abs 확인 / venue: Comments 없음, Anthology에도 없음
- **Vision-DeepResearch Benchmark (VDR-Bench)** (arXiv [2602.02185](https://arxiv.org/abs/2602.02185), Yu Zeng et al., v1 2026-02-02)
  - 학회: **Preprint (arXiv)**. EMNLP 2026 Main과 Findings 볼륨에 이 제목이 없다. 현재 저자 README 제목에도 "[ICML 2026 & EMNLP 2026]" 표기가 없다.
  - VQA 2,000개로 구성되며, 다단계 큐레이션과 전문가 검토를 거쳤다. 기존 벤치마크가 시각 검색 중심이 아니고, 이미지는 거의 정확히 일치하는 검색만으로 풀리는 등 평가 시나리오가 지나치게 이상적이라고 지적한다. 다회차 crop 검색 워크플로도 제안했다.
  - 검증: ✅ arXiv abs 확인. 1차 노트의 "10개 시각 도메인", "multi-hop"은 초록에 없다. / venue: 확인 불가

### Inferences
- 채택이 1차 출처로 확인된 논문만 보면 학회 지형이 바뀌었다. 2025년에는 ML/CV 학회(ICLR·CVPR·NeurIPS)와 ACL 2025가 중심이었고, 2026년에는 NLP 학회(ACL 2026 Main/Findings, EMNLP 2026 Main/Findings) 비중이 크다. 멀티모달 RAG가 검색 표현 연구에서 추론·에이전트 연구로 바뀌어 가는 흐름과 맞는다. 다만 NeurIPS 2026(ODE, VL-DocIR, arXiv Comments 기준)과 ACM MM 2026(Hyper-M2RAG, arXiv Comments 기준) 채택 사례도 있어서, NLP 학회로만 쏠린 것은 아니다.
- ColPali, VisRAG, VDocRAG는 이후 거의 모든 문서 RAG 논문의 비교 기준선이자 인용 출발점이다. 기초 논문으로 표시해 포함할 만하다.
- 2026년 10월 이후 arXiv 논문은 대부분 아직 채택 정보가 없다. 예외는 VL-DocIR(NeurIPS 2026)이다. 나머지는 "Preprint (arXiv)"로 적는 것이 맞다.

### Gaps
- EVisRAG(ACM MM 2026)와 Vision-DeepResearch(ICML 2026)의 채택 여부는 여전히 저자 공지로만 확인된다. ACM DL, ICML proceedings, OpenReview에 접속할 수 없어 **Unverified**로 남겼다.
- ICLR·NeurIPS·CVPR 채택은 arXiv Comments나 PDF 헤더로 확인했다. 학회 측 1차 페이지(OpenReview 결정문, iclr.cc, neurips.cc, CVF)는 이번에도 접속이 차단되어 교차 확인하지 못했다.
- 1차 Gaps에서 채택 학회를 모른다고 했던 논문들은 이번에 다음과 같이 확인되었다.
  - Video-RAG ([2411.13093](https://arxiv.org/abs/2411.13093), Yongdong Luo et al.): arXiv Comments에 "Accepted at NeurIPS 2025. Camera-ready version"이 있어 **NeurIPS 2025**다.
  - VideoRAG ([2501.05874](https://arxiv.org/abs/2501.05874), Soyeong Jeong et al.): **ACL 2025 Findings**다 ([Anthology 2025.findings-acl.1096](https://aclanthology.org/2025.findings-acl.1096/)).
  - VideoRAG ([2502.01549](https://arxiv.org/abs/2502.01549), Xubin Ren et al.): v1뿐이고 PDF에 "Preprint. Under review." 표기가 있어 **Preprint (arXiv)**다.
  - M3DocRAG ([2411.04952](https://arxiv.org/abs/2411.04952), Jaemin Cho et al.): Comments에 학회가 없고 ACL/EMNLP 볼륨에도 없다. **Unverified**다.
  - ViDoRAG(EMNLP 2025 Main)와 UniversalRAG(ACL 2026 Main)는 위 보조 후보 항목에서 확인했다.

## Q2. 이 하위 분야는 어떻게 세분화되는가 (문서/이미지 RAG, 비디오 RAG, 오디오 RAG, 멀티모달 에이전트, 벤치마크)?

### Takeaway
서베이의 분류 축(도메인, 검색 모달리티, 검색 단위, 그래프·에이전트 확장)과 실제 논문 분포를 합쳐 보면 여섯 갈래로 나누는 것이 적절하다. (A) 시각 문서 RAG, (B) 이미지·지식 기반 VQA RAG, (C) 비디오 RAG, (D) 오디오·음성 RAG, (E) 멀티모달 검색·딥리서치 에이전트, (F) 벤치마크·서베이. (A)는 다시 검색기, 파이프라인, 증거 선택·추론, 에이전트로 나뉜다.

### Cited Findings
- 문서 이해용 멀티모달 RAG 서베이는 도메인, 검색 모달리티, 검색 단위로 분류 체계를 세우고, 그래프 구조와 에이전트 프레임워크 관련 진전을 함께 리뷰한다 — [arXiv 초록](https://arxiv.org/abs/2510.15253), [ACL Anthology](https://aclanthology.org/2026.acl-long.204/)
- "Ask in Any Modality" 서베이 저장소는 검색 전략을 Text-Centric, Vision-Centric, Video-Centric, Audio-Centric으로 나누고, 데이터셋도 Image-Text, Video-Text, Audio-Text, Medical, Fashion으로 분류한다 — [GitHub](https://github.com/llm-lab-org/Multimodal-RAG-Survey) (README 분류. 논문 초록은 텍스트·이미지·오디오·비디오를 포괄한다고만 적고 있다: [arXiv](https://arxiv.org/abs/2502.08826))
- **제안 분류와 논문 배치 (괄호 안은 2차 검증한 venue):**
  - **A. 시각 문서 RAG (페이지를 이미지로 검색)**
    - A1 검색기·인덱싱: ColPali [F1] (ICLR 2025) — [arXiv](https://arxiv.org/abs/2407.01449)
    - A2 VLM 기반 end-to-end 파이프라인: VisRAG [F2] (ICLR 2025), VDocRAG [F3] (CVPR 2025) — [arXiv](https://arxiv.org/abs/2410.10594), [arXiv](https://arxiv.org/abs/2504.09795)
    - A3 다중 이미지 증거 선택·추론·압축: EVisRAG [4] (venue Unverified), Utility-Oriented [5] (ACL 2026 Main), Doc-REFRAG [6] (EMNLP 2026 Main) — [arXiv](https://arxiv.org/abs/2510.09733), [ACL](https://aclanthology.org/2026.acl-long.1620/), [ACL](https://aclanthology.org/2026.emnlp-main.377/)
    - A4 에이전트·RL 문서 RAG: ViDoRAG (EMNLP 2025 Main), MDocRAG-RL (ACL 2026 Findings), VRAG-RL (Preprint), VLD-RAG (Preprint, arXiv 2607.24748: 키워드 희소 검색과 dense 의미 질의를 결합하고, Retrieval·Answer·Validation 에이전트가 검증 중심 워크플로로 협업) — [ACL](https://aclanthology.org/2025.emnlp-main.464/), [ACL](https://aclanthology.org/2026.findings-acl.420/), [arXiv](https://arxiv.org/abs/2505.22019), [arXiv](https://arxiv.org/abs/2607.24748)
  - **B. 이미지·지식 기반 VQA와 범용 멀티모달 RAG:** Utility-Oriented [5] (MRAG-Bench와 Visual-RAG로 평가), UniversalRAG (ACL 2026 Main, 모달리티별 코퍼스로 라우팅), MR-RAG (의료 VQA, CVPR 2026 ⚠️) — [ACL](https://aclanthology.org/2026.acl-long.1620/), [ACL](https://aclanthology.org/2026.acl-long.177/), [CVPR](https://cvpr.thecvf.com/virtual/2026/poster/36898)
  - **C. 비디오 RAG·장편 비디오 이해:**
    - Vgent [7] (NeurIPS 2025 Spotlight)
    - Rethinking RAG in Long Videos (2606.13141, Preprint)
    - Event-Causal RAG (2605.06185, Preprint): 고정 길이 구간 검색이 사건을 조각내는 문제를 다룬다.
    - Coarse Indexing, Fine Evidence (2608.23011, Preprint): DAGC로 인덱싱 시간 단위와 증거 시간 단위를 분리한다.
    - 근거: [arXiv](https://arxiv.org/abs/2510.14032), [arXiv](https://arxiv.org/abs/2606.13141), [arXiv](https://arxiv.org/abs/2605.06185), [arXiv](https://arxiv.org/abs/2608.23011)
  - **D. 오디오·음성 RAG:** WavRAG [8] (ACL 2025 Main), PlanRAG-Audio [9] (ACL 2026 Findings), HARP (2609.14116, arXiv Comments "SLT 2026"), AudioHelix (Hierarchical Dual-Graph Retrieval for Long-form Audio Reasoning, 2610.41390, Preprint) — [ACL](https://aclanthology.org/2025.acl-long.613/), [ACL](https://aclanthology.org/2026.findings-acl.1304/), [arXiv](https://arxiv.org/abs/2609.14116), [arXiv](https://arxiv.org/abs/2610.41390)
  - **E. 멀티모달 검색·딥리서치 에이전트:** MMSearch-R1 [10] (ACL 2026 Main), Vision-DeepResearch [11] (venue Unverified), Video-DeepResearch (2608.03979, EMNLP 2026 Findings), ODE (2605.10832, NeurIPS 2026), Long-MDR (2610.05195), OneSearch-VL (2610.12419), DeepVoyager-VL (2608.01827), SimpleSearch-VL (2606.31504). 마지막 넷은 Preprint다 — [ACL](https://aclanthology.org/2026.acl-long.114/), [arXiv](https://arxiv.org/abs/2601.22060), [ACL](https://aclanthology.org/2026.findings-emnlp.1881/), [arXiv](https://arxiv.org/abs/2605.10832), [arXiv](https://arxiv.org/abs/2610.05195), [arXiv](https://arxiv.org/abs/2610.12419)
  - **F. 벤치마크·서베이:** Scaling Beyond Context [12] (ACL 2026 Main), Ask in Any Modality (ACL 2025 Findings), MRAG-Bench (ICLR 2025), VDR-Bench (2602.02185, Preprint), V-RAGBench (2606.13141 안에 포함), VisBrowse-Bench (2603.16289, Preprint), VistaHop (2606.03273, Preprint), VL-DocIR (2610.17057, arXiv Comments "Accepted by NeurIPS 2026") — [ACL](https://aclanthology.org/2026.acl-long.204/), [ACL](https://aclanthology.org/2025.findings-acl.861/), [arXiv](https://arxiv.org/abs/2410.08182), [arXiv](https://arxiv.org/abs/2602.02185), [arXiv](https://arxiv.org/abs/2603.16289), [arXiv](https://arxiv.org/abs/2606.03273), [arXiv](https://arxiv.org/abs/2610.17057)

#### Q2·Q3에만 나오는 논문 검증 (모두 arxiv.org/abs 직접 확인)
- **VLD-RAG** [2607.24748](https://arxiv.org/abs/2607.24748). 첫 저자 Seonok Kim. v1 2026-05-21(제출 이력 기준, ID는 2607). 검증: ✅ arXiv abs / venue: Preprint (arXiv). 1차 노트의 "모달리티 간 일관성 제약"은 초록에 없어 검증 에이전트 워크플로 설명으로 바꿨다.
- **Event-Causal RAG (EC-RAG)** [2605.06185](https://arxiv.org/abs/2605.06185). Peizheng Yan. v1 2026-05-07. 검증: ✅ arXiv abs. 고정 구간 검색이 사건을 조각낸다는 문제의식, ECV-1H 벤치마크(1,251 QA), 4.96–11.67% 향상을 확인했다. / venue: Preprint (arXiv).
- **Coarse Indexing, Fine Evidence (DAGC)** [2608.23011](https://arxiv.org/abs/2608.23011). Zhe Jin. v1 2026-08-24. 검증: ✅ arXiv abs. 그래프 노드를 약 40–50%만 유지하면서 1.3–1.7배 가속하고, QA 성능은 약 99%를 유지한다. / venue: Preprint (arXiv).
- **HARP: Agentic Hybrid Retrieval and Analysis for Long-Form Audio** [2609.14116](https://arxiv.org/abs/2609.14116). Chin-Jou Li. v1 2026-09-12. 검증: ✅ arXiv abs / venue: **SLT 2026** (arXiv Comments 기준. 1차 노트는 프리프린트로 적었다).
- **Hierarchical Dual-Graph Retrieval for Long-form Audio Reasoning (AudioHelix)** [2610.41390](https://arxiv.org/abs/2610.41390). Eunsang Lee. v1 2026-10-30. 검증: ✅ arXiv abs / venue: Preprint (arXiv).
- **Video-DeepResearch** [2608.03979](https://arxiv.org/abs/2608.03979). Zhen Fang. v1 2026-08-04. 검증: ✅ arXiv abs와 Anthology / venue: **EMNLP 2026 Findings** ([ACL Anthology 2026.findings-emnlp.1881](https://aclanthology.org/2026.findings-emnlp.1881/)).
- **Towards On-Policy Data Evolution for Visual-Native Multimodal Deep Search Agents (ODE)** [2605.10832](https://arxiv.org/abs/2605.10832). Shijue Huang. v1 2026-05-11. 검증: ✅ arXiv abs / venue: **NeurIPS 2026** (arXiv Comments 기준. 1차 노트의 "프리프린트"를 정정했다).
- **Long-MDR** [2610.05195](https://arxiv.org/abs/2610.05195). On Tai Tang. v1 2026-10-04. 검증: ✅ arXiv abs. 128k 컨텍스트와 75턴 이상의 도구 상호작용을 확인했다. / venue: Preprint (arXiv).
- **OneSearch-VL** [2610.12419](https://arxiv.org/abs/2610.12419). Hongyu Li. v1 2026-10-08. 검증: ✅ arXiv abs / venue: Preprint (arXiv).
- **DeepVoyager-VL** [2608.01827](https://arxiv.org/abs/2608.01827). Huanyao Zhang. v1 2026-08-03. 검증: ✅ arXiv abs. 주의할 점은 이 논문이 **RL 없이 합성 데이터 SFT만** 쓴다는 것이다. / venue: Preprint (arXiv).
- **SimpleSearch-VL** [2606.31504](https://arxiv.org/abs/2606.31504). Ming Dai. v1 2026-06-30. 검증: ✅ arXiv abs. arXiv Comments는 "Technical Report"다. / venue: Preprint (arXiv).
- **VisBrowse-Bench** [2603.16289](https://arxiv.org/abs/2603.16289). Zhengbo Zhang. v1 2026-03-17. 검증: ✅ arXiv abs. VQA 169개이고, 최고 모델 Claude-4.6-Opus의 정확도가 47.6%였다. / venue: Preprint (arXiv).
- **VistaHop** [2606.03273](https://arxiv.org/abs/2606.03273). Hang He. v1 2026-06-02. 검증: ✅ arXiv abs. 과제 600개이고, 최고 모델의 Pass@1이 26.33%였다. / venue: Preprint (arXiv).
- **VL-DocIR** [2610.17057](https://arxiv.org/abs/2610.17057). Simon Ott. v1 2026-10-12. 검증: ✅ arXiv abs / venue: **NeurIPS 2026** (arXiv Comments 기준).
- **MegaRAG: Multimodal Knowledge Graph-Based RAG** [2512.20626](https://arxiv.org/abs/2512.20626). Chi-Hsiang Hsiao. v1 2025-11-26. 검증: ✅ arXiv abs와 Anthology / venue: **ACL 2026 Main** ([ACL Anthology 2026.acl-long.2218](https://aclanthology.org/2026.acl-long.2218/)).
- **Multimodal Graph RAG for Long-range Visually Rich Document Understanding** [2606.28780](https://arxiv.org/abs/2606.28780). Yi-Cheng Wang. v1 2026-06-27. 검증: ⚠️ arXiv abs는 확인했다. 그러나 1차 노트가 붙인 이름 "KG4VD"와 "질의별 검색 모드 선택"은 제목과 초록에 없다. DLVQA 벤치마크는 확인했다. / venue: **WACV 2027** (arXiv Comments 기준).
- **Hypergraph-based Multimodal RAG with Incremental Refinement (Hyper-M2RAG)** [2608.16628](https://arxiv.org/abs/2608.16628). Shenao Chen. v1 2026-08-17. 검증: ✅ arXiv abs / venue: **ACM MM 2026** (arXiv Comments "Accepted to the 34th ACM International Conference on Multimedia").
- **Does More Retrieved Evidence Help Visual Retrieval-Augmented Generation with Diffusion Language Models?** [2608.07006](https://arxiv.org/abs/2608.07006). Jiankun Wang. v1 2026-08-07. 검증: ✅ arXiv abs. 제목은 "…with Diffusion Language Models"까지가 전체이고, 대상은 확산 언어모델(DLM)에 한정된다. / venue: Preprint (arXiv).

### Inferences
- 학회 채택이 확인된 논문만 세면 A(문서)와 E(에이전트)가 가장 두껍고, D(오디오)는 아직 얇다. 오디오 쪽 2026년 채택 사례로는 PlanRAG-Audio(ACL 2026 Findings)와 HARP(SLT 2026, arXiv Comments 기준)를 확인했다.
- 그래프 기반 멀티모달 RAG는 A와 C에 걸쳐 늘고 있다. MegaRAG(ACL 2026 Main), 2606.28780(WACV 2027), Hyper-M2RAG(ACM MM 2026), Vgent(NeurIPS 2025), DAGC(Preprint)가 그 예다. 다만 이번 과제 범위에서는 그래프 RAG를 다른 연구자가 맡으므로 경계 사례로만 표시한다 — [arXiv](https://arxiv.org/abs/2512.20626), [arXiv](https://arxiv.org/abs/2606.28780), [arXiv](https://arxiv.org/abs/2608.16628)

### Gaps
- 위 분류는 서베이 두 편의 축에 이번 조사 결과를 합쳐 연구자가 직접 만든 것이다. 하위 분야별 논문 수 같은 정량 통계는 확보하지 못했다.

## Q3. 2026년에 새로 떠오른 방향은 무엇인가?

### Takeaway
2026년 흐름은 다섯 가지로 요약된다.
1. 수십 단계, 수백 번 검색하는 장기(long-horizon) 멀티모달 딥리서치 에이전트. 대부분 RL로 학습하지만 DeepVoyager-VL은 SFT만 쓴다. 대상을 비디오까지 넓히는 시도도 있다.
2. "더 많이 검색"이 아니라 증거의 효용을 고르고 시각 토큰을 압축·확장하는 효율적 다중 이미지 RAG.
3. 시각 환각 억제.
4. 장시간 오디오·장편 비디오에서 검색 단위와 계획을 다시 설계하는 연구.
5. 시각 중심 브라우징·딥서치 벤치마크의 급증.

### Cited Findings
- **장기 멀티모달 딥리서치:**
  - Vision-DeepResearch는 추론 단계 수십 회와 검색엔진 상호작용 수백 회를 지원한다 — [arXiv](https://arxiv.org/abs/2601.22060). ICML 2026 채택은 저자 repo 공지뿐이라 Unverified다 — [GitHub](https://github.com/Osilly/Vision-DeepResearch).
  - Long-MDR은 멀티모달 리서치 에이전트의 온라인 RL을 128k 컨텍스트, 75턴 이상의 도구 상호작용까지 늘렸다 — [arXiv](https://arxiv.org/abs/2610.05195).
  - ODE(On-policy Data Evolution)는 멀티모달 딥서치 벤치마크 8종 평균에서 Qwen3-VL-8B 에이전트를 24.9%에서 39.0%로, 30B를 30.6%에서 41.5%로 올렸다 — [arXiv](https://arxiv.org/abs/2605.10832). arXiv Comments 기준 NeurIPS 2026 채택이다.
- **이미지에서 비디오로 확장되는 딥리서치:**
  - Video-DeepResearch(v1 2026-08-04)는 멀티모달 에이전트를 정적 이미지에서 연속 비디오 스트림으로 확장했다. 이 설정에서는 dense 시공간 grounding과 오픈 웹 탐색이 함께 필요하다. 35B-A3B 모델이 Video-DR-Bench 평균 64.0%를 기록했다 — [arXiv](https://arxiv.org/abs/2608.03979). 학회는 EMNLP 2026 Findings다 — [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.1881/). 저자 repo는 이 논문을 "Video-DR extends Vision-DR to continuous video streams"라고 소개한다 — [GitHub](https://github.com/Osilly/Vision-DeepResearch).
  - OneSearch-VL은 단일 이미지·다중 이미지·비디오 딥리서치를 하나로 다루는 에이전트다. Visually Grounded Evidence Graph(VGEG)로 시각 앵커, 개체 관계, 출처가 있는 사실 사이의 의존성을 보존한다 — [arXiv](https://arxiv.org/abs/2610.12419).
- **효용 중심 증거 선택과 효율:**
  - 유사도 대신 정보 이득으로 증거 효용을 추정하는 training-free 방법(ACL 2026 Main) — [ACL Anthology](https://aclanthology.org/2026.acl-long.1620/)
  - 질문당 평균 37.4장을 검색하는 설정에서 compress-and-expand로 지연을 낮춘 Doc-REFRAG(EMNLP 2026 Main) — [ACL Anthology](https://aclanthology.org/2026.emnlp-main.377/)
  - 관련 문제 제기로 "Does More Retrieved Evidence Help Visual RAG with Diffusion Language Models?"(2608.07006)가 있다. DLM에서는 검색 페이지를 늘리면 정답 페이지 recall은 오르지만, 모두 넘기면 의미 충돌 때문에 정확도가 자주 떨어진다 — [arXiv](https://arxiv.org/abs/2608.07006)
- **시각 환각 억제:** VisRAG 2.0은 v2(2026-07-28)에서 제목에 "Mitigating Visual Hallucinations"를 넣었다. 초록도 환각 감소와 평균 약 19% 향상을 내세운다 — [arXiv](https://arxiv.org/abs/2510.09733). ACM MM 2026 채택은 저자 공지뿐이다 — [GitHub](https://github.com/OpenBMB/VisRAG).
- **장시간 오디오 RAG:** PlanRAG-Audio(ACL 2026 Findings)는 먼저 계획한 뒤 검색하는 방식을 제시했다. 이후 HARP(2609.14116, SLT 2026)는 키워드와 벡터를 결합한 하이브리드 검색에 메타데이터와 오디오 증거를 함께 쓰면 단일 모달리티보다 답 정확도가 약 10% 오른다고 보고했다. AudioHelix(2610.41390)는 시간적으로 떨어진 텍스트·음향 증거를 계층적 이중 그래프로 연결한다 — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1304/), [arXiv](https://arxiv.org/abs/2609.14116), [arXiv](https://arxiv.org/abs/2610.41390)
- **장편 비디오 RAG의 단위·평가 재설계:**
  - V-RAGBench는 검색과 생성을 따로 평가하는 벤치마크이고, CARVE는 청크 단위로 설정을 적응적으로 고른다 — [arXiv](https://arxiv.org/abs/2606.13141)
  - EC-RAG는 사건 단위 인과 검색을 하며 ECV-1H 벤치마크를 함께 제안했다 — [arXiv](https://arxiv.org/abs/2605.06185)
  - DAGC는 인덱싱 단위와 증거 단위를 분리한다 — [arXiv](https://arxiv.org/abs/2608.23011)
- **시각 중심 딥서치 벤치마크:**
  - VDR-Bench(VQA 2,000개)는 기존 벤치마크가 시각 검색 중심이 아니고 평가 시나리오를 지나치게 이상적으로 가정한다고 지적한다 — [arXiv](https://arxiv.org/abs/2602.02185)
  - VisBrowse-Bench(169개)와 VistaHop(600개)은 각각 시각 네이티브 브라우징 검색과 장기 시각 딥서치를 평가한다 — [arXiv](https://arxiv.org/abs/2603.16289), [arXiv](https://arxiv.org/abs/2606.03273)
- **문서 RAG의 에이전트화·그래프화:**
  - VLD-RAG: 희소 키워드 검색과 dense 의미 질의를 결합하고, Retrieval·Answer·Validation 에이전트가 검증 중심으로 협업한다. Preprint — [arXiv](https://arxiv.org/abs/2607.24748)
  - Multimodal Graph RAG for Long-range VRD (2606.28780): 문서 단위 멀티모달 KG를 자동으로 구축하고 DLVQA 벤치마크를 제안했다. WACV 2027 — [arXiv](https://arxiv.org/abs/2606.28780)

### Inferences
- 2025년의 대표 질문은 "텍스트로 파싱할 것인가, 이미지로 검색할 것인가"였다. 2026년에는 "검색된 수십 장의 이미지 중 무엇을, 얼마나 정밀하게 볼 것인가", 그리고 "검색 자체를 정책으로 학습할 것인가"로 옮겨갔다.
- 멀티모달 딥리서치·검색 에이전트 논문은 2026년에 급증했지만 대부분 프리프린트다. 1차 출처로 채택이 확인된 것은 다음과 같다.
  - MMSearch-R1: ACL 2026 Main, Anthology
  - Video-DeepResearch: EMNLP 2026 Findings, Anthology
  - ODE: NeurIPS 2026, arXiv Comments
  - Vision-DeepResearch의 ICML 2026은 저자 공지뿐이다.

### Gaps
- Neos, Pix2Evidence 등 이름만 언급된 2026년 10–11월 프리프린트는 arXiv ID가 노트에 없어 이번 검증에 포함하지 않았다. 초록과 채택 여부는 미확인이다.
- 1차 노트에서 "한 후속 연구가 PlanRAG-Audio보다 6.0–32.8점 높다"고 한 주장은 HARP와 AudioHelix 어느 쪽 초록에도 없다. 출처를 확인하지 못해 Findings에서 뺐다(**Unverified**).
- 1차 노트의 V-RAGBench 수치(영상 216개, 11–99시간, triplet 2,100개)는 arXiv 초록에 없다. 제3자 요약에서 온 수치라 **Unverified**이고 Findings에서 뺐다.
- ViDoRe V3 등 ColPali 계열의 2026년 후속 벤치마크는 찾지 못했다.
- CVPR 2026과 ICCV 계열에서 비디오 RAG 채택 논문은 체계적으로 조사하지 못했다. CVF와 cvpr.thecvf.com이 프록시로 차단되어 있기 때문이다.

## Verification log
2026-11-19 2차 검증. 1차 소스는 arxiv.org/abs와 PDF, aclanthology.org 논문 페이지와 볼륨 목록이다. OpenReview, iclr.cc, neurips.cc, icml.cc, CVF, PMLR은 접속이 차단되거나 챌린지가 걸렸다.

- **존재 확인:** 노트의 arXiv ID 42개를 모두 arxiv.org/abs에서 확인했다. 핵심 12편, 보조 후보, Q1 Gaps, Q2·Q3 언급 논문이 모두 포함된다. 제목과 저자가 맞지 않는 ID는 없었고, **삭제한 논문은 없다.** arXiv ID가 없는 두 편은 이렇게 처리했다.
  - MDocRAG-RL: ACL Anthology에서 존재를 확인했다.
  - MR-RAG: arXiv 제목 검색 결과가 0건이다. CVF 포스터 페이지는 검색 색인으로만 확인되었고 직접 받지 못해 ⚠️로 유지했다.
- **첫 저자 정정·확인**
  - EVisRAG: 미확인에서 Yubo Sun으로 정정했다.
  - WavRAG: "Chen et al."에서 Yifu Chen으로 정정했다.
  - MDocRAG-RL: Zhongyu Wang을 새로 확인했다.
- **v1 날짜 정정**
  - ColPali: 2024-07에서 2024-06-27로 고쳤다.
  - EVisRAG: 2025-10-14에서 2025-10-10으로 고쳤다.
  - MMSearch-R1: 2025-06-26에서 2025-06-25로 고쳤다.
  - VDocRAG(04-14), WavRAG(02-20), Utility(2026-05-13), Doc-REFRAG(2026-08-31), Vision-DR(2026-01-29)는 월 단위에서 일 단위로 구체화했다.
- **제목:** EVisRAG의 v1과 v2 제목을 구분해 적었다. 2608.07006은 전체 제목을 복원했다("…with Diffusion Language Models"). 2606.28780은 "KG4VD"라는 이름이 arXiv 제목과 초록에 없다고 명시했다.
- **venue 정정·추가 (Unverified나 프리프린트에서 확인된 상태로 바뀐 것)**
  - UniversalRAG: ACL 2026 Main (Anthology 2026.acl-long.177)
  - ViDoRAG: EMNLP 2025 Main (Anthology 2025.emnlp-main.464)
  - MDocRAG-RL: ACL 2026 Findings (Anthology 2026.findings-acl.420)
  - Video-DeepResearch: EMNLP 2026 Findings (Anthology 2026.findings-emnlp.1881)
  - ODE: NeurIPS 2026 (arXiv Comments)
  - HARP: SLT 2026 (arXiv Comments)
  - MegaRAG: ACL 2026 Main (Anthology 2026.acl-long.2218)
  - Hyper-M2RAG: ACM MM 2026 (arXiv Comments)
  - 2606.28780: WACV 2027 (arXiv Comments)
  - VL-DocIR: NeurIPS 2026 (arXiv Comments)
  - Video-RAG(2411.13093): NeurIPS 2025 (arXiv Comments)
  - VideoRAG(2501.05874): ACL 2025 Findings (Anthology 2025.findings-acl.1096)
  - Ask in Any Modality: 저자 공지에서 Anthology 2025.findings-acl.861로 근거를 바꿨다.
- **venue 재확인 (변경 없음)**
  - ColPali, MRAG-Bench: ICLR 2025 (arXiv Comments)
  - VisRAG: ICLR 2025 (PDF v2 헤더)
  - VDocRAG: CVPR 2025 (arXiv Comments)
  - Vgent: NeurIPS 2025 Spotlight (arXiv Comments)
  - WavRAG: ACL 2025 Main
  - Utility-Oriented, MMSearch-R1, Scaling Beyond Context: ACL 2026 Main
  - PlanRAG-Audio: ACL 2026 Findings
  - Doc-REFRAG: EMNLP 2026 Main
  - 위의 Anthology 근거는 모두 해당 Anthology 페이지와 페이지 번호로 확인했다.
- **여전히 Unverified인 venue**
  - EVisRAG ACM MM 2026, Vision-DeepResearch ICML 2026: 저자 repo 공지뿐이다.
  - VRAG-RL, VimRAG, VDR-Bench, M3DocRAG: 1차 출처 어디에도 채택 정보가 없다. VDR-Bench는 EMNLP 2026 Main·Findings 볼륨에도 없었다.
  - MR-RAG CVPR 2026: CVF를 직접 확인하지 못했다.
- **요약·수치 정정 (초록과 대조)**
  - VisRAG: 25–39%(v1)와 20–40%(v2)로 버전 차이를 명시했다.
  - EVisRAG: 27%(v1)와 약 19%(v2)를 명시하고 환각 감소를 추가했다.
  - MMSearch-R1: "검색 30% 감소"는 더 큰 RAG 모델과 비교한 수치로 고쳤다. "7B·FVQA 공개"는 뺐다.
  - Vgent: "training-free 4단계", "2B–7B", "LongVideoBench·Video-MME"를 뺐다. 3.0–5.4%는 MLVU 기준임을 명시했다.
  - WavRAG: 10x로 확정하고 "5–14배" 언급을 지웠다.
  - PlanRAG-Audio: "zero-shot·수작업 SQL 없음"을 뺐다.
  - Vision-DR: "8B·30B-A3B", "6개 벤치마크 SOTA", "최초 장기 MLLM"을 뺐다.
  - Scaling Beyond Context: "첫 서베이" 주장을 뺐다. 20–200M 토큰 인용은 HTML v2 본문에서 원문을 확인했다.
  - Utility: "Top-K만 대형 모델로"를 뺐고, 이론적 동치성을 추가했다.
  - Doc-REFRAG: "디코딩 중 원해상도 확장" 표현을 초록 표현으로 바꿨다.
  - ViDoRAG: "actor-critic"을 GMM 하이브리드 검색과 탐색·요약·성찰 워크플로로 바꿨다.
  - VLD-RAG: "모달리티 간 일관성 제약"을 Retrieval·Answer·Validation 에이전트 워크플로로 바꿨다.
  - VDR-Bench: "10개 도메인, multi-hop"은 초록에 없다고 표시했다.
- **Findings에서 뺀 주장 (출처 미확인):** "후속 연구가 PlanRAG-Audio보다 6.0–32.8점 우위", V-RAGBench의 영상 216개·11–99시간·triplet 2,100개 수치.
- **인용구:** 모든 인용구를 arXiv 초록, Anthology 초록, arXiv HTML 본문 원문과 대조했다. 맞지 않던 것은 원문 문장으로 바꿨다(EVisRAG, MMSearch-R1, Vision-DR의 쉼표 표기).
