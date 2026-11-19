# Multimodal RAG: 최신 논문 큐레이션 (기준일 2026-11-19)

> **검증 방법 및 한계 (보고서 작성자 필독)**
> - 이 세션에서는 arxiv.org, export.arxiv.org, openreview.net, aclanthology.org, CVF 사이트에 직접 접속할 수 없었다(DNS/프록시 차단). 그래서 학회 채택 여부는 다음 두 경로로 확인했다. (1) 검색엔진이 색인한 1차 페이지 스니펫: ACL Anthology 항목, iclr.cc/neurips.cc/cvpr.thecvf.com 포스터 페이지, CVF Open Access, arXiv "Comments" 필드. (2) `raw.githubusercontent.com`에서 직접 받은 저자 공식 GitHub README의 "accepted" 공지.
> - 출처가 **저자 저장소 공지뿐인** 항목은 표에 "(저자 repo 공지 기준)"이라고 따로 표시했다.
> - 인용구는 검색 스니펫이나 저자 README에 그대로 나온 문장에서만 골랐다. 원문과 직접 대조하지 못한 인용구는 따로 표시했다.
> - 날짜는 출처가 확인된 경우 일(日) 단위로 적었다. 그렇지 않으면 arXiv ID의 YYMM을 기준으로 월 단위까지만 적었다.

## Q1. 가장 영향력 있는 논문과 최신(2025–2026) Multimodal RAG 논문은 무엇이며, 각각 어느 학회에 채택되었나?

### Takeaway
이 분야의 출발점은 페이지를 이미지로 검색하는 문서 RAG다(ColPali·VisRAG는 ICLR 2025, VDocRAG는 CVPR 2025). 2025년 하반기부터 2026년 사이에는 세 갈래로 넓어졌다. (a) 다중 이미지 증거를 고르고 추론하는 연구(ACL 2026 Main, EMNLP 2026 Main, ACM MM 2026), (b) 비디오·오디오 RAG(NeurIPS 2025, ACL 2025, ACL 2026 Findings), (c) RL로 학습한 멀티모달 검색·딥리서치 에이전트(MMSearch-R1은 ACL 2026 Main, Vision-DeepResearch는 ICML 2026). 2026년 최신 논문 중 상당수는 아직 arXiv 프리프린트다.

### Cited Findings

#### 핵심 큐레이션 목록 (12편)

**[F1] ColPali: Efficient Document Retrieval with Vision Language Models** (기초 논문)
- 저자: Manuel Faysse et al.
- arXiv: [2407.01449](https://arxiv.org/abs/2407.01449). 최초 공개: 2024-07 (arXiv ID 기준)
- 학회: **ICLR 2025**. 근거: [ICLR 2025 포스터 페이지](https://iclr.cc/virtual/2025/poster/28336), arXiv PDF 헤더 "Published as a conference paper at ICLR 2025" ([arXiv PDF](https://arxiv.org/pdf/2407.01449))
- 요약:
  1. 시각적으로 복잡한 문서 검색을 평가하는 페이지 단위 벤치마크 ViDoRe를 제안했다. 또 VLM으로 페이지 이미지를 텍스트 추출 없이 바로 multi-vector 임베딩하는 late-interaction 검색기 ColPali를 제시했다.
  2. 기존 OCR·파싱 기반 문서 검색 파이프라인보다 성능이 크게 높으면서도 더 단순하고 빠르며 end-to-end로 학습할 수 있다. 모델·데이터·코드·벤치마크는 hf.co/vidore에 공개되어 있다.
- 인용: "largely outperforms modern document retrieval pipelines while being drastically simpler, faster and end-to-end trainable" ([arXiv](https://arxiv.org/abs/2407.01449))

**[F2] VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents** (기초 논문)
- 저자: Shi Yu et al.
- arXiv: [2410.10594](https://arxiv.org/abs/2410.10594). 최초 공개: 2024-10-14 ([저자 repo News](https://github.com/OpenBMB/VisRAG))
- 학회: **ICLR 2025** (Poster). 근거: [ICLR 2025 포스터 페이지](https://iclr.cc/virtual/2025/poster/27679), OpenReview ID zG459X3Xge ([papers.cool의 OpenReview 미러](https://papers.cool/venue/zG459X3Xge@OpenReview))
- 요약:
  1. 문서를 텍스트로 파싱하는 대신 VLM으로 페이지를 이미지 그대로 임베딩·검색하고, 검색된 이미지로 VLM이 답을 생성하는 방식이다. 이렇게 해서 파싱 과정의 정보 손실을 없앴다. 검색기 학습에는 공개 데이터와 합성 데이터를 함께 썼다.
  2. 텍스트 기반 RAG보다 검색과 생성 양쪽에서 성능이 높았다. 검색 스니펫에 따르면 end-to-end 성능이 25–39% 향상되었다 ([ICLR 슬라이드/요약](https://iclr.cc/virtual/2025/poster/27679)). 버전마다 수치 표기가 다를 수 있으므로 원문 확인이 필요하다.
- 인용: "the document is directly embedded using a VLM as an image and then retrieved to enhance the generation of a VLM" ([ICLR 포스터](https://iclr.cc/virtual/2025/poster/27679))

**[F3] VDocRAG: Retrieval-Augmented Generation over Visually-Rich Documents** (기초 논문, 2025 상반기)
- 저자: Ryota Tanaka et al. (NTT)
- arXiv: [2504.09795](https://arxiv.org/abs/2504.09795). 최초 공개: 2025-04 (arXiv ID 기준)
- 학회: **CVPR 2025**. 근거: [CVF Open Access, pp. 24827–24837](https://openaccess.thecvf.com/content/CVPR2025/html/Tanaka_VDocRAG_Retrieval-Augmented_Generation_over_Visually-Rich_Documents_CVPR_2025_paper.html), [저자 repo "accepted to CVPR 2025"](https://github.com/nttmdlab-nlp/VDocRAG)
- 요약:
  1. 차트·표가 섞인 PDF·PPTX 문서 코퍼스를 하나의 이미지 형식으로 다룬다. 시각 정보를 dense 토큰으로 압축하면서 텍스트와 정렬하는 self-supervised 사전학습 과제를 만들어, LVLM을 검색기로 적응시켰다.
  2. 텍스트 기반 RAG보다 성능이 크게 높고 일반화 능력도 강했다. 개방 도메인 문서 VQA 데이터셋을 처음으로 통합한 OpenDocVQA도 공개했다.
- 인용: "the first unified collection of open-domain document visual question answering datasets" ([CVPR 2025 포스터](https://cvpr.thecvf.com/virtual/2025/poster/34926))

**[4] VisRAG 2.0 / EVisRAG: Evidence-Guided Multi-Image Reasoning in Visual Retrieval-Augmented Generation**
- 최신 버전 제목: "VisRAG2.0: Mitigating Visual Hallucinations via Evidence-Guided Multi-Image Reasoning in Visual Retrieval-Augmented Generation" ([arXiv HTML v2](https://arxiv.org/html/2510.09733v2))
- 저자: 첫 저자명은 직접 확인하지 못했다. OpenBMB VisRAG 팀 논문이고 Shi Yu가 공저자다. ADS bibcode 2025arXiv251009733S를 보면 첫 저자 성이 'S'로 시작한다 ([ADS](https://ui.adsabs.harvard.edu/abs/2025arXiv251009733S/abstract)).
- arXiv: [2510.09733](https://arxiv.org/abs/2510.09733). 최초 공개: 2025-10-14 ([저자 repo News](https://github.com/OpenBMB/VisRAG))
- 학회: **ACM MM 2026** (저자 repo 공지 기준). repo의 "20260709: Our work EVisRAG is accepted by ACM MM 2026" 공지를 근거로 했다 ([GitHub](https://github.com/OpenBMB/VisRAG)). ACM DL에서는 확인하지 못했다.
- 요약:
  1. 기존 Visual RAG는 여러 검색 이미지에 걸친 증거를 제대로 지각·통합하지 못해 근거가 약하고 결론이 틀리는 문제가 있다. EVisRAG는 이미지별 증거를 먼저 기록하고 그것을 모아 답하는 end-to-end 모델이다. 학습에는 토큰 범위별로 세분화된 보상을 주는 RS-GRPO를 제안했다.
  2. 여러 VQA 벤치마크에서 백본 VLM보다 크게 향상되었다. v1 초록은 평균 27% 향상, alphaXiv 저자 페이지는 약 19% 향상과 시각적 환각 감소를 적고 있어 수치가 서로 다르다 ([검색 결과](https://arxiv.org/abs/2510.09733)). 최종 버전 수치는 원문 확인이 필요하다.
- 인용: "binds fine-grained rewards to scope-specific tokens to jointly optimize visual perception and reasoning abilities of VLMs" ([저자 README, 초록과 동일 문장](https://github.com/OpenBMB/VisRAG))

**[5] Utility-Oriented Visual Evidence Selection for Multimodal Retrieval-Augmented Generation**
- 저자: Weiqing Luo et al.
- arXiv: [2605.13277](https://arxiv.org/abs/2605.13277). 최초 공개: 2026-05 (arXiv ID 기준)
- 학회: **ACL 2026 Main** (long paper). 근거: [ACL Anthology 2026.acl-long.1620](https://aclanthology.org/2026.acl-long.1620/), [저자 repo "[ACL 2026]"](https://github.com/Hcnaeg/utility-mrag)
- 요약:
  1. 시각 증거 선택을 의미적 유사도가 아니라 정보이론 문제로 다룬다. 증거의 효용을 모델 출력 분포에 대한 정보 이득으로 정의했다. 작은 surrogate MLLM이 각 이미지의 "helpfulness"를 매기고, Top-K만 큰 모델에 넘기는 training-free 방식이다.
  2. MRAG-Bench와 Visual-RAG에서 최신 RAG 기준선보다 일관되게 높은 성능을 보였고, 계산 비용도 크게 줄였다.
- 인용: "existing methods typically rely on semantic relevance or surface-level similarity" ([ACL Anthology](https://aclanthology.org/2026.acl-long.1620/))

**[6] Doc-REFRAG: Rethinking Multimodal Document Retrieval-Augmented Generation**
- 저자: Ruofan Hu et al. (Zhou Zhao 교신)
- arXiv: [2608.30163](https://arxiv.org/abs/2608.30163). 최초 공개: 2026-08 (arXiv ID 기준)
- 학회: **EMNLP 2026 Main**. 근거: arXiv Comments "Accepted by EMNLP 2026 Main", [ACL Anthology 2026.emnlp-main.377](https://aclanthology.org/2026.emnlp-main.377/)
- 요약:
  1. 검색된 문서 이미지가 수십 장에 이르는 다중 이미지 RAG에서, 시각 토큰을 정적으로 잘라내는 대신 compress-and-expand 방식을 쓴다. 이미지를 거친 청크 임베딩으로 압축해 두고, RL로 학습한 선택기가 질문과 관련된 청크만 디코딩 중에 원해상도 토큰으로 다시 펼친다. 학습 데이터는 DocLongRAG(343K QA, 질문당 평균 37.4장)이다.
  2. 6개 벤치마크에서 11개 강력한 기준선을 넘는 SOTA 정확도를 기록했고, 추론 지연은 더 낮았다.
- 인용: "Real-world knowledge resides in multimodal documents, necessitating retrieval-augmented generation (RAG) for accurate question answering." ([arXiv](https://arxiv.org/abs/2608.30163))

**[7] Vgent: Graph-based Retrieval-Reasoning-Augmented Generation For Long Video Understanding**
- 저자: Xiaoqian Shen et al. (Mohamed Elhoseiny 교신)
- arXiv: [2510.14032](https://arxiv.org/abs/2510.14032). 최초 공개: 2025-10-15
- 학회: **NeurIPS 2025 (Spotlight)**. 근거: [NeurIPS 2025 포스터 페이지](https://neurips.cc/virtual/2025/loc/san-diego/poster/119823), [저자 repo "[NeurIPS 2025 Spotlight]"](https://github.com/xiaoqian-shen/Vgent)
- 요약:
  1. 장편 비디오를 오프라인에서 클립 그래프로 구조화한다. 질의 키워드로 그래프에서 클립을 검색하고, 검색된 클립 위에서 구조화된 추론을 거쳐 답을 생성한다. 학습이 필요 없는 4단계 RAG 프레임워크다.
  2. 2B–7B 규모의 여러 LVLM에서 3.0–5.4% 향상되었고, 기존 RAG 기반 비디오 이해 방법보다 8.6% 높았다. 평가는 MLVU, LongVideoBench, Video-MME 등 장편 비디오 벤치마크 3종으로 했다.
- 인용: "a novel graph-based retrieval-reasoning-augmented generation framework to enhance LVLMs for long video understanding" ([NeurIPS 포스터](https://neurips.cc/virtual/2025/loc/san-diego/poster/119823))

**[8] WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models**
- 저자: Chen et al. (ACL Anthology 표기). 첫 저자 이름 전체는 원문에서 확인하지 못했다.
- arXiv: [2502.14727](https://arxiv.org/abs/2502.14727). 최초 공개: 2025-02 (arXiv ID 기준)
- 학회: **ACL 2025 Main** (long paper). 근거: [ACL Anthology 2025.acl-long.613](https://aclanthology.org/2025.acl-long.613/)
- 요약:
  1. ASR로 음성을 텍스트로 바꾸지 않고 원시 오디오를 바로 임베딩·검색한다. 텍스트와 오디오가 섞인 지식베이스에서 교차 모달 검색도 지원하는 음성 대화 모델용 RAG다.
  2. ASR 후 텍스트로 처리하는 RAG와 비슷한 검색 성능을 내면서 속도는 약 10배 빠르다. 다만 같은 arXiv 목록의 다른 부분에는 5–14배로 적혀 있어 표기가 일치하지 않는다 ([검색 결과](https://huggingface.co/papers/2502.14727)).
- 인용: "the first retrieval augmented generation framework with native, end-to-end audio support" ([arXiv/HF](https://huggingface.co/papers/2502.14727))

**[9] PlanRAG-Audio: Planning and Retrieval Augmented Generation for Long-form Audio Understanding**
- 저자: Masao Someki et al. (Shinji Watanabe 교신)
- arXiv: [2605.20414](https://arxiv.org/abs/2605.20414). 최초 공개: 2026-05-19
- 학회: **ACL 2026 Findings**. 근거: arXiv Comments "Accepted to Findings of ACL 2026", [ACL Anthology 2026.findings-acl.1304](https://aclanthology.org/2026.findings-acl.1304/) (pp. 26167–26183)
- 요약:
  1. 장시간 오디오에서는 입력 길이가 길고 발화 내용·화자·감정·음향 이벤트 단서가 시간축에 흩어져 있다. 이를 구조화된 검색 문제로 바꿨다. 먼저 질의에 필요한 모달리티와 시간 구간을 계획하고, 텍스트·오디오 DB에서 그 부분만 가져온다.
  2. 추론 정확도가 높아졌고 오디오가 길어져도 성능이 안정적으로 유지되었다. 추론 비용이 원시 오디오 길이에 비례해 늘지 않고, 작업별 프롬프트나 수작업 SQL 없이 zero-shot으로 동작한다.
- 인용(검색 스니펫 기반, 원문 대조 필요): "a planning-based retrieval-augmented generation framework for scalable long-form audio understanding" ([arXiv](https://arxiv.org/abs/2605.20414))

**[10] MMSearch-R1: Incentivizing LMMs to Search**
- 저자: Jinming Wu et al. (Anthology 표기 "Wu et al.")
- arXiv: [2506.20670](https://arxiv.org/abs/2506.20670). 최초 공개: 2025-06-26 ([저자 repo News](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1))
- 학회: **ACL 2026 Main**. 근거: [ACL Anthology 2026.acl-long.114](https://aclanthology.org/2026.acl-long.114/), [저자 repo "[26.04.07] accepted to ACL-2026 Main conference"](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)
- 요약:
  1. 고정된 RAG 파이프라인이나 프롬프트 기반 검색 에이전트가 과도하게 검색하는 문제를 다룬다. LMM이 언제·무엇을 검색할지 스스로 정하도록 이미지 검색과 텍스트 검색 도구를 붙여 end-to-end RL로 학습했다. 검색이 필요한 예와 필요 없는 예를 균형 있게 섞은 데이터셋과, 검색 페널티 보상을 함께 사용했다.
  2. 지식 집약형 VQA에서 같은 크기의 RAG 기준선보다 정확도가 높았고, 검색 호출은 30% 이상 줄었다. 7B 모델과 FVQA 데이터셋을 공개했다.
- 인용: "on-demand, multi-turn search" ([저자 README/초록](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1))

**[11] Vision-DeepResearch: Incentivizing DeepResearch Capability in Multimodal Large Language Models**
- 저자: Wenxuan Huang et al.
- arXiv: [2601.22060](https://arxiv.org/abs/2601.22060). 최초 공개: 2026-01 (arXiv ID 기준)
- 학회: **ICML 2026** (저자 repo 공지 기준). repo의 "[2026/05/01] Vision-DeepResearch has been accepted by ICML 2026"을 근거로 했다 ([GitHub](https://github.com/Osilly/Vision-DeepResearch)). ICML proceedings에서는 확인하지 못했다.
- 요약:
  1. 다회차·다개체·다중 스케일의 시각·텍스트 검색을 수행하는 멀티모달 딥리서치 패러다임을 제안했다. cold-start SFT와 RL로 딥리서치 능력을 모델 안에 내재화했다. 추론 단계는 수십 회, 검색엔진 상호작용은 수백 회 규모로 늘렸고, 저자들은 이를 최초의 장기(long-horizon) 멀티모달 딥리서치 MLLM이라고 부른다.
  2. 8B와 30B-A3B 모델이 사실 중심 VQA 벤치마크 6종에서 SOTA를 기록했다. GPT-5, Gemini-2.5-Pro, Claude-4-Sonnet 기반 딥리서치 시스템보다도 높았다.
- 인용: "multi-turn, multi-entity, and multi-scale visual and textual search" ([HF Papers](https://huggingface.co/papers/2601.22060))

**[12] Scaling Beyond Context: A Survey of Multimodal Retrieval-Augmented Generation for Document Understanding** (서베이)
- 저자: Sensen Gao et al. (Mingming Gong 교신)
- arXiv: [2510.15253](https://arxiv.org/abs/2510.15253). 최초 공개: 2025-10-17 (v2는 2026-01-09)
- 학회: **ACL 2026 Main**. 근거: [ACL Anthology 2026.acl-long.204](https://aclanthology.org/2026.acl-long.204/), 저자 repo "[ACL2026 Main]" ([GitHub](https://github.com/SensenGao/Multimodal-RAG-Survey-For-Document))
- 요약:
  1. 문서 이해용 멀티모달 RAG를 세 축으로 분류했다: 도메인(open/closed), 검색 모달리티(image/text/hybrid), 검색 단위(page/element). 그래프 인덱싱과 에이전트 오케스트레이션 같은 하이브리드 확장도 함께 정리했다.
  2. 데이터셋·벤치마크·평가 지표를 정리하고, 효율성·세밀한 표현·강건성을 남은 과제로 꼽았다. 저자들은 멀티모달 RAG와 문서 이해를 명시적으로 연결한 첫 서베이라고 설명한다.
- 인용(검색 스니펫 요약 수준, 원문 대조 필요): 멀티모달 RAG 벤치마크에는 약 20–200M 시각 토큰이 필요한데, 기존 MLLM의 컨텍스트는 128K–1M에 그친다고 지적한다 ([arXiv HTML v2](https://arxiv.org/html/2510.15253v2)).

#### 보조 후보 (채택 상태 포함, 보고서에 선택적으로 사용)
- **Ask in Any Modality: A Comprehensive Survey on Multimodal RAG** (arXiv 2502.08826): **ACL 2025 Findings** (저자 repo 공지 "accepted for publication in the ACL 2025 Findings"). 이미지·비디오·오디오 중심 검색과 에이전트형 상호작용을 다루는 포괄적 서베이다 — [GitHub](https://github.com/llm-lab-org/Multimodal-RAG-Survey)
- **MRAG-Bench** (비전 중심 RAG 벤치마크): **ICLR 2025** (저자 repo 공지). README에 따르면 LVLM은 텍스트 지식보다 이미지로 보강했을 때 더 크게 향상된다 — [GitHub](https://github.com/mragbench/MRAG-Bench)
- **UniversalRAG** (arXiv 2504.20734): 모달리티·단위가 서로 다른 여러 코퍼스 중 알맞은 곳으로 질의를 라우팅하는 RAG다. OpenReview forum(mXtV92SPxW)은 있으나 채택 학회는 **Unverified** — [arXiv](https://arxiv.org/abs/2504.20734), [OpenReview](https://openreview.net/forum?id=mXtV92SPxW)
- **VRAG-RL** (arXiv 2505.22019)와 **VimRAG** (arXiv 2602.12735, 텍스트·이미지·비디오 대상 멀티모달 메모리 그래프와 GGPO): Alibaba-NLP 저장소 README에서 채택 공지를 찾지 못해 **Unverified** — [GitHub](https://github.com/Alibaba-NLP/VRAG)
- **ViDoRAG** (arXiv 2502.18017, 다중 에이전트 actor-critic 방식, ViDoSeek 벤치마크): README에 채택 공지가 없어 **Unverified** — [GitHub](https://github.com/Alibaba-NLP/ViDoRAG)
- **MR-RAG** (의료 VQA, 관련성 인식 멀티모달 RAG): **CVPR 2026** 포스터. 도메인 특화 논문이고, 세 의료 데이터셋에서 기준선보다 최대 6.4% 향상 — [CVPR 2026 포스터](https://cvpr.thecvf.com/virtual/2026/poster/36898)
- **MDocRAG-RL** (RL 기반 멀티모달 문서 RAG): 다른 논문의 참고문헌에서만 ACL 2026 Findings(doi 10.18653/v1/2026.findings-acl.420)로 인용된 것을 보았다. 1차 페이지는 확인하지 못해 **Unverified** — [검색 결과 출처](https://aclanthology.org/2026.acl-long.204.pdf)
- **Rethinking RAG in Long Videos: What to Retrieve and How to Use It?** (arXiv 2606.13141): V-RAGBench(검색과 생성을 따로 평가)와 CARVE(청크별로 적응적 재순위화)를 제안했다. **Preprint (arXiv)**, 채택 정보 없음 — [arXiv](https://arxiv.org/abs/2606.13141)
- **Vision-DeepResearch Benchmark (VDR-Bench)** (arXiv 2602.02185): 10개 시각 도메인의 multi-hop VQA 2,000개로 구성된다. Vision-DR repo 제목에 "[ICML 2026 & EMNLP 2026]"이라고 되어 있지만 벤치마크의 EMNLP 채택은 1차 출처로 확인하지 못해 **Unverified** — [arXiv](https://arxiv.org/abs/2602.02185), [GitHub](https://github.com/Osilly/Vision-DeepResearch)

### Inferences
- 채택이 확인된 논문만 보면 학회 지형이 2025년 ML/CV 학회(ICLR·CVPR·NeurIPS)에서 2026년 NLP 학회(ACL 2026 Main/Findings, EMNLP 2026 Main) 쪽으로 옮겨갔다. 멀티모달 RAG가 검색 표현 연구에서 추론·에이전트 연구로 바뀌어 가는 흐름과 맞는다.
- ColPali, VisRAG, VDocRAG는 이후 거의 모든 문서 RAG 논문의 비교 기준선이자 인용 출발점이다. 기초 논문으로 표시해 포함할 만하다.
- 2026년 하반기 arXiv 논문(10월 이후)은 대부분 아직 채택 정보가 없다. ICLR 2027이나 CVPR 2027에서 심사 중일 가능성이 높으므로 "Preprint"로 적는 것이 맞다.

### Gaps
- arxiv.org와 openreview.net에 직접 접속하지 못해 EVisRAG(ACM MM 2026)와 Vision-DeepResearch(ICML 2026)의 채택 여부는 저자 공지로만 확인했다. 공식 proceedings에서는 확인하지 못했다.
- EVisRAG와 WavRAG의 첫 저자 전체 이름은 원문에서 확인하지 못했다.
- ColPali, VDocRAG, WavRAG, Utility, Doc-REFRAG, Vision-DR은 arXiv v1의 정확한 날짜(일 단위)를 확인하지 못해 ID 기준으로 월만 적었다.
- PlanRAG-Audio와 Scaling Beyond Context 서베이는 초록 원문을 그대로 확보하지 못해 인용구를 원문과 대조할 필요가 있다.
- Video-RAG (2411.13093), VideoRAG (2501.05874, 2502.01549), M3DocRAG (2411.04952), ViDoRAG (2502.18017), UniversalRAG의 채택 학회는 이번 조사에서 확인하지 못했다.

## Q2. 이 하위 분야는 어떻게 세분화되는가 (문서/이미지 RAG, 비디오 RAG, 오디오 RAG, 멀티모달 에이전트, 벤치마크)?

### Takeaway
서베이의 분류 축(도메인, 검색 모달리티, 검색 단위, 그래프·에이전트 확장)과 실제 논문 분포를 합쳐 보면 여섯 갈래로 나누는 것이 적절하다. (A) 시각 문서 RAG, (B) 이미지·지식 기반 VQA RAG, (C) 비디오 RAG, (D) 오디오·음성 RAG, (E) 멀티모달 검색·딥리서치 에이전트, (F) 벤치마크·서베이. (A)는 다시 검색기, 파이프라인, 증거 선택·추론으로 나뉜다.

### Cited Findings
- 문서 이해용 멀티모달 RAG 서베이는 방법을 도메인(open/closed), 검색 모달리티(image/text/hybrid), 검색 단위(page/element)로 나누고, 그래프 인덱싱과 에이전트 오케스트레이션을 하이브리드 확장으로 따로 분류한다 — [Scaling Beyond Context 프로젝트 페이지](https://sensengao.github.io/Multimodal-RAG-Survey/), [ACL Anthology](https://aclanthology.org/2026.acl-long.204/)
- "Ask in Any Modality" 서베이 저장소는 검색 전략을 모달리티 중심으로 Text-Centric, Vision-Centric, Video-Centric, Audio-Centric으로 나누고, 데이터셋도 Image-Text, Video-Text, Audio-Text, Medical, Fashion으로 분류한다 — [GitHub](https://github.com/llm-lab-org/Multimodal-RAG-Survey)
- **제안 분류와 논문 배치:**
  - **A. 시각 문서 RAG (페이지를 이미지로 검색)**
    - A1 검색기·인덱싱: ColPali [F1] — [arXiv](https://arxiv.org/abs/2407.01449)
    - A2 VLM 기반 end-to-end 파이프라인: VisRAG [F2], VDocRAG [F3] — [ICLR](https://iclr.cc/virtual/2025/poster/27679), [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Tanaka_VDocRAG_Retrieval-Augmented_Generation_over_Visually-Rich_Documents_CVPR_2025_paper.html)
    - A3 다중 이미지 증거 선택·추론·압축: EVisRAG [4], Utility-Oriented [5], Doc-REFRAG [6] — [arXiv](https://arxiv.org/abs/2510.09733), [ACL](https://aclanthology.org/2026.acl-long.1620/), [ACL](https://aclanthology.org/2026.emnlp-main.377/)
    - A4 에이전트·RL 문서 RAG (프리프린트 위주): VRAG-RL, ViDoRAG, VLD-RAG (arXiv 2607.24748, 희소 키워드 검색과 dense 시각 검색 결합) — [GitHub](https://github.com/Alibaba-NLP/VRAG), [arXiv](https://arxiv.org/abs/2607.24748)
  - **B. 이미지·지식 기반 VQA와 범용 멀티모달 RAG:** Utility-Oriented [5] (MRAG-Bench, Visual-RAG 평가), UniversalRAG (모달리티별 코퍼스로 라우팅), MR-RAG (의료 VQA, CVPR 2026) — [ACL](https://aclanthology.org/2026.acl-long.1620/), [arXiv](https://arxiv.org/abs/2504.20734), [CVPR](https://cvpr.thecvf.com/virtual/2026/poster/36898)
  - **C. 비디오 RAG·장편 비디오 이해:** Vgent [7], Rethinking RAG in Long Videos (2606.13141), Event-Causal RAG (2605.06185, 고정 길이 구간 검색이 사건을 쪼개는 문제), Coarse Indexing, Fine Evidence (2608.23011, 인덱싱 시간 단위 분리) — [NeurIPS](https://neurips.cc/virtual/2025/loc/san-diego/poster/119823), [arXiv](https://arxiv.org/abs/2606.13141), [arXiv](https://arxiv.org/abs/2605.06185), [arXiv](https://arxiv.org/pdf/2608.23011)
  - **D. 오디오·음성 RAG:** WavRAG [8], PlanRAG-Audio [9], HARP (2609.14116), Hierarchical Dual-Graph Retrieval for Long-form Audio (2610.41390) — [ACL](https://aclanthology.org/2025.acl-long.613/), [ACL](https://aclanthology.org/2026.findings-acl.1304/), [arXiv](https://arxiv.org/html/2609.14116), [arXiv](https://arxiv.org/html/2610.41390)
  - **E. 멀티모달 검색·딥리서치 에이전트:** MMSearch-R1 [10], Vision-DeepResearch [11], Video-DeepResearch (2608.03979), Long-MDR (2610.05195), OneSearch-VL (2610.12419), DeepVoyager-VL (2608.01827), SimpleSearch-VL (2606.31504) — [ACL](https://aclanthology.org/2026.acl-long.114/), [GitHub](https://github.com/Osilly/Vision-DeepResearch), [arXiv](https://arxiv.org/html/2610.05195), [arXiv](https://arxiv.org/pdf/2610.12419)
  - **F. 벤치마크·서베이:** Scaling Beyond Context [12] (ACL 2026 Main), Ask in Any Modality (ACL 2025 Findings), MRAG-Bench (ICLR 2025), VDR-Bench (2602.02185), V-RAGBench (2606.13141 안에 포함), VisBrowse-Bench (2603.16289), VistaHop (2606.03273), VL-DocIR (2610.17057) — [ACL](https://aclanthology.org/2026.acl-long.204/), [GitHub](https://github.com/mragbench/MRAG-Bench), [arXiv](https://arxiv.org/abs/2602.02185), [arXiv](https://arxiv.org/pdf/2603.16289), [arXiv](https://arxiv.org/pdf/2606.03273)

### Inferences
- 학회 채택이 확인된 논문만 세면 A(문서)와 E(에이전트)가 가장 두껍고, D(오디오)는 아직 얇다. 오디오 쪽 2026년 채택 사례로는 PlanRAG-Audio(Findings)만 확인했다.
- 그래프 기반 멀티모달 RAG(MegaRAG 2512.20626, KG4VD 2606.28780, Hypergraph MRAG 2608.16628)는 A와 C에 걸쳐 늘고 있다. 다만 이번 과제 범위에서는 그래프 RAG를 다른 연구자가 맡으므로 경계 사례로만 표시한다 — [arXiv](https://arxiv.org/html/2512.20626v2), [arXiv](https://arxiv.org/abs/2606.28780)

### Gaps
- 위 분류는 서베이 두 편의 축에 이번 조사 결과를 합쳐 연구자가 직접 만든 것이다. 하위 분야별 논문 수 같은 정량 통계는 확보하지 못했다.

## Q3. 2026년에 새로 떠오른 방향은 무엇인가?

### Takeaway
2026년 흐름은 다섯 가지로 요약된다. ① RL로 학습해 수십 단계, 수백 번 검색하는 장기(long-horizon) 멀티모달 딥리서치 에이전트와, 그 대상을 비디오까지 넓히는 시도. ② "더 많이 검색"이 아니라 증거의 효용을 고르고 시각 토큰을 압축·확장하는 효율적 다중 이미지 RAG. ③ 시각 환각 억제. ④ 장시간 오디오·장편 비디오에서 검색 단위와 계획을 다시 설계하는 연구. ⑤ 시각 중심 브라우징·딥서치 벤치마크의 급증.

### Cited Findings
- **장기 RL 멀티모달 딥리서치:** Vision-DeepResearch는 추론 단계 수십 회, 검색엔진 상호작용 수백 회 규모를 내세우며 ICML 2026에 채택되었다(저자 repo 공지) — [GitHub](https://github.com/Osilly/Vision-DeepResearch). Long-MDR은 멀티모달 리서치 에이전트의 온라인 RL을 128k 컨텍스트, 75회 이상 도구 상호작용까지 늘렸다 — [arXiv](https://arxiv.org/html/2610.05195). ODE(On-policy Data Evolution)는 Qwen3-VL-8B 에이전트의 성능을 멀티모달 딥서치 벤치마크 8종 평균 24.9%에서 39.0%로 올렸다 — [arXiv](https://arxiv.org/html/2605.10832) (프리프린트)
- **이미지에서 비디오로 확장되는 딥리서치:** Video-DeepResearch(2026-08)는 Vision-DR을 연속 비디오 스트림으로 확장했다 — [저자 repo News](https://github.com/Osilly/Vision-DeepResearch). OneSearch-VL은 이미지와 비디오를 함께 다루는 통합 딥리서치 에이전트로, 비디오 리서치에는 시공간 grounding이 필요하다고 본다 — [arXiv](https://arxiv.org/pdf/2610.12419)
- **효용 중심 증거 선택과 효율:** 유사도 대신 정보 이득으로 효용을 계산해 Top-K 증거를 고르는 방법(ACL 2026 Main) — [ACL Anthology](https://aclanthology.org/2026.acl-long.1620/). 질문당 평균 37.4장을 검색하는 설정에서 compress-and-expand로 지연을 낮춘 Doc-REFRAG(EMNLP 2026 Main) — [ACL Anthology](https://aclanthology.org/2026.emnlp-main.377/). 관련 문제 제기로 "Does More Retrieved Evidence Help Visual RAG…"(2608.07006)가 있다 — [arXiv](https://arxiv.org/pdf/2608.07006)
- **시각 환각 억제:** VisRAG 2.0은 v2에서 제목을 "Mitigating Visual Hallucinations via Evidence-Guided Multi-Image Reasoning"으로 바꾸고 환각 완화를 전면에 내세웠다(ACM MM 2026, 저자 공지) — [arXiv HTML v2](https://arxiv.org/html/2510.09733v2), [GitHub](https://github.com/OpenBMB/VisRAG)
- **장시간 오디오 RAG:** PlanRAG-Audio(ACL 2026 Findings)가 먼저 계획한 뒤 검색하는 방식을 제시한 이후, HARP(2609.14116)와 Hierarchical Dual-Graph(2610.41390) 같은 후속 프리프린트가 나왔다. 한 후속 연구는 PlanRAG-Audio보다 6.0–32.8점 높다고 보고했다 — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1304/), [arXiv](https://arxiv.org/html/2609.14116)
- **장편 비디오 RAG의 단위·평가 재설계:** 검색과 생성을 따로 평가하는 V-RAGBench(216개 영상, 11–99시간 분량, 2,100개 triplet; 수치는 제3자 요약 기준). 사건 단위 인과 검색(EC-RAG). 인덱싱 단위와 증거 단위를 분리하는 연구(2608.23011) — [arXiv](https://arxiv.org/abs/2606.13141), [arXiv](https://arxiv.org/abs/2605.06185), [arXiv](https://arxiv.org/pdf/2608.23011)
- **시각 중심 딥서치 벤치마크:** VDR-Bench(2,000개 multi-hop VQA, 10개 도메인)는 기존 벤치마크가 시각 검색 중심이 아니고 검색 조건을 지나치게 이상적으로 가정한다고 지적한다. VisBrowse-Bench와 VistaHop은 시각 중심 브라우징과 장기 시각 딥서치를 평가한다 — [arXiv](https://arxiv.org/abs/2602.02185), [arXiv](https://arxiv.org/pdf/2603.16289), [arXiv](https://arxiv.org/pdf/2606.03273)
- **문서 RAG의 에이전트화·그래프화:** VLD-RAG(희소+dense 시각 검색, 모달리티 간 일관성 제약), KG4VD(질의별 검색 모드 선택, DLVQA 데이터셋) — [arXiv](https://arxiv.org/abs/2607.24748), [arXiv](https://arxiv.org/abs/2606.28780) (둘 다 프리프린트)

### Inferences
- 2025년의 대표 질문이 "텍스트로 파싱할 것인가, 이미지로 검색할 것인가"였다면, 2026년에는 "검색된 수십 장의 이미지 중 무엇을, 얼마나 정밀하게 볼 것인가", 그리고 "검색 자체를 정책으로 학습할 것인가"로 옮겨갔다.
- 멀티모달 딥리서치 에이전트 논문은 2026년에 급증했지만 대부분 프리프린트다. 채택이 확인된 대표작은 MMSearch-R1(ACL 2026 Main)과 Vision-DeepResearch(ICML 2026, 저자 공지)뿐이다.

### Gaps
- 2026년 10–11월 프리프린트(Long-MDR, OneSearch-VL, Neos, Pix2Evidence 등)의 초록 원문과 채택 여부는 확인하지 못했다. ICLR 2027 심사 중일 가능성이 있으나 확인되지 않았다.
- ViDoRe V3 등 ColPali 계열 2026년 후속 벤치마크는 검색으로 찾지 못했다.
- CVPR 2026과 ICCV 계열에서 비디오 RAG 채택 논문을 체계적으로 조사하지 못했다. 공식 목록 사이트에 접속할 수 없었기 때문이다.
