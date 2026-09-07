# 🎯 면접 코치 AI — 이형 스타일 피드백 시스템

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Gemini-Flash_Lite-orange?logo=google" />
  <img src="https://img.shields.io/badge/ChromaDB-0.6.3-green" />
  <img src="https://img.shields.io/badge/RAG-기반-purple" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

> **"면접왕 이형"** 유튜브 채널의 면접 노하우를 학습한 AI가  
> 직설적이고 구체적인 면접 피드백을 생성합니다.

---

## 💡 프로젝트 소개

취업 준비생이 면접 답변을 입력하면, AI가 **대기업 인사담당자 출신 면접 코치 '이형' 스타일**로 피드백을 제공합니다.

- 단순한 "잘했어요/못했어요"가 아닌 **왜 좋은지/나쁜지** 면접관 시각으로 분석
- 이형 유튜브 영상 74개 + 실제 면접 인터뷰·피드백 데이터를 RAG 지식베이스로 활용
- 음성·표정·자세 분석 모듈로 **비언어 피드백**까지 확장 가능

---

## 🗂️ 프로젝트 구조

```
interview-coach/
│
├── 📓 01_rag_build.ipynb          # RAG 지식베이스 구축 (ChromaDB)
├── 📓 02_eval_pipeline.ipynb      # 자동 평가 파이프라인 (LLM-as-judge)
├── 📓 03_feedback.ipynb           # ✅ 메인 피드백 생성기
├── 📓 gemini_튜닝.ipynb            # Gemini API 파인튜닝
├── 🐍 05_chunk_experiment.py      # 청킹 파라미터 실험
│
├── 📁 data/
│   ├── rag_data/                  # 이형 유튜브 강의 정리 텍스트 (74개, 5개 카테고리)
│   ├── feedback_data/             # 실제 면접 인터뷰 + 피드백 쌍 데이터 (97개)
│   └── finetune/
│       ├── gemini_train.jsonl     # Gemini 파인튜닝 데이터 (68 examples)
│       └── openai_train.jsonl     # OpenAI 파인튜닝 데이터
│
├── 📁 data_prep/
│   └── LLM용JSON.ipynb            # 파인튜닝 데이터 생성 스크립트
│
└── 📁 models/
    ├── llm_comparison/            # LLM 선정 비교 실험 (GPT-4.1 mini vs Gemini 2.5 Pro)
    └── nonverbal/                 # 비언어 분석 (음성·표정·자세)
        ├── whisper_test.ipynb
        ├── crepe_test.ipynb
        ├── silero_test.ipynb
        ├── 표정분석.ipynb
        ├── DeepFace.ipynb
        ├── 고개_각도_측정.ipynb
        └── 몸_움직임_측정.ipynb
```

---

## 🤖 피드백 형식 — 이형 스타일 3섹션

```
[이형의 팩폭 한줄평]
한 문장으로 이 답변의 핵심 문제 또는 강점을 직격한다.

[이형의 시선]
면접관 관점에서 이 답변이 어떻게 들리는지, 왜 좋은지/나쁜지 구체적으로 분석한다.
(3~5문장)

[이형의 합격 처방전]
1. 즉시 실천 가능한 구체적 개선 방법
2. 답변 구조/내용 개선 방법
3. 면접관에게 어필할 포인트
```

---

## 🏗️ 시스템 아키텍처

```
면접 질문 + 지원자 답변
        │
        ▼
  ┌─────────────┐     ┌──────────────────────────────┐
  │  ChromaDB   │────▶│  관련 청크 Top-4 검색          │
  │  (RAG DB)   │     │  paraphrase-multilingual-     │
  │  786 chunks │     │  MiniLM-L12-v2 (384차원)      │
  └─────────────┘     └──────────────┬───────────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │   Gemini Flash Lite     │
                        │   이형 스타일 프롬프트    │
                        └────────────┬───────────┘
                                     │
                                     ▼
                           이형 스타일 피드백 출력
                      (팩폭 한줄평 / 시선 / 합격 처방전)
```

---

## 🛠️ 기술 스택

| 구성 요소 | 기술 | 비고 |
|---|---|---|
| **LLM** | Gemini Flash Lite | 무료, 1500회/일 |
| **벡터 DB** | ChromaDB 0.6.3 | 로컬 영구 저장 |
| **임베딩** | paraphrase-multilingual-MiniLM-L12-v2 | 한국어 지원, 384차원 |
| **청킹** | 300자 / overlap 50자 | 문장 경계 분할 |
| **평가** | LLM-as-judge | Gemini가 Gemini 출력 평가 |
| **비언어** | Whisper / DeepFace / MediaPipe | 음성·표정·자세 |

---

## 📊 평가 메트릭

| 메트릭 | 설명 | 범위 |
|---|---|---|
| `factpunch` | 팩폭 한줄평의 날카로움 | 1–5 |
| `analysis` | 시선 분석의 깊이 | 1–5 |
| `prescription` | 처방전의 실천 가능성 | 1–5 |
| `accuracy` | RAG 근거 활용도 | 1–5 |
| `context_relevance` | 검색 청크 관련성 | 0–1 |
| `keyword_hit_rate` | 핵심 키워드 포함율 | 0–1 |

---

## 📦 데이터셋

### RAG 지식베이스 (74개 텍스트)
| 카테고리 | 파일 수 | 내용 |
|---|---|---|
| 핵심 역량 & 취업 전략 | 12개 | 필살기 구조, 지원동기, 포트폴리오 |
| 비즈니스 커뮤니케이션 | 8개 | 두괄식 말하기, 면접관 언어 |
| 면접 단계별 공략 | 12개 | 1분 자기소개, PT면접, 다대일면접 |
| 필수·압박 질문 방어 | 10개 | 퇴직사유, 장단점, 입사 후 포부 |
| 면접 마인드셋 | 9개 | 멘탈 관리, 면접 복기, 합격자 특징 |

### 파인튜닝 데이터
- `gemini_train.jsonl` — 68 examples (이형 3섹션 형식)
- `openai_train.jsonl` — GPT-4.1 mini 파인튜닝 완료 데이터

---

## 🚀 빠른 시작

### 1. 환경 설정
```bash
git clone https://github.com/whdyddls4276/interview_coach.git
cd interview_coach
pip install -r requirements.txt
cp .env.example .env
# .env 파일에 GOOGLE_API_KEY 입력
```

### 2. RAG 지식베이스 구축
```
01_rag_build.ipynb 실행
→ chroma_db/ 폴더 생성 (786개 청크)
```

### 3. 피드백 받기
```
03_feedback.ipynb 실행
→ STEP 4 셀에서 질문 / 답변 입력 후 실행
```

### 4. 성능 평가
```
02_eval_pipeline.ipynb 실행
→ 6개 메트릭 자동 측정
```

---

## ⚠️ API 키 설정

`.env` 파일을 생성하고 아래와 같이 입력하세요:
```
GOOGLE_API_KEY=your_google_api_key_here
```
> API 키는 절대 코드에 직접 넣지 마세요. `.gitignore`에 `.env`가 포함되어 있습니다.

---

## 🔬 LLM 비교 실험 결과

`models/llm_comparison/` 폴더에서 3가지 LLM을 비교 실험했습니다:

| 모델 | 방식 | 비고 |
|---|---|---|
| Gemini Flash Lite | RAG + 프롬프트 | ✅ 최종 채택 (무료) |
| GPT-4.1 mini | 파인튜닝 | 유료, 파인튜닝 완료 |
| Gemini 2.5 Pro | Vertex AI context | 유료, Colab 기반 |

---

## 📌 향후 계획

- [ ] 비언어 분석 (음성·표정·자세) 메인 파이프라인에 통합
- [ ] 웹 UI (Streamlit) 연동
- [ ] RAG 데이터 확장 (현재 74개 → 목표 200개+)
- [ ] Gemini 파인튜닝 (Vertex AI 활용)
