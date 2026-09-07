# 🎯 면접 코치 AI — 이형 스타일 피드백 시스템

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" />
  <img src="https://img.shields.io/badge/Gemini-3.1_Pro-orange?logo=google" />
  <img src="https://img.shields.io/badge/ChromaDB-0.6.3-green" />
  <img src="https://img.shields.io/badge/RAG_+_Few--Shot-purple" />
  <img src="https://img.shields.io/badge/Multimodal-Video-red" />
</p>

> **"면접왕 이형"** 채널의 면접 노하우를 학습한 AI가
> 직설적이고 구체적인 면접 피드백을 생성합니다.
> **면접 영상을 넣으면 답변 내용 + 비언어적 태도를 함께 분석합니다.**

---

## 💡 프로젝트 소개

취업 준비생이 면접 답변을 **텍스트 또는 영상**으로 입력하면, AI가 대기업 인사담당자 출신 면접 코치 '이형' 스타일로 피드백을 제공합니다.

- 단순한 "잘했어요/못했어요"가 아닌 **왜 좋은지/나쁜지** 면접관 시각으로 분석
- **영상 입력 시** 시선·표정·자세 등 비언어 요소까지 `[MM:SS]` 타임스탬프로 지적
- 답변 내용과 태도가 **불일치하는 지점**을 짚어냄

---

## 🧩 핵심 설계 — 데이터의 두 가지 역할

이 프로젝트의 핵심은 두 데이터셋을 **서로 다른 목적**으로 나눠 쓴다는 점입니다.

| 데이터 | 역할 | 해결하는 문제 |
|---|---|---|
| `data/feedback_data/` | **Few-Shot** | 이형 특유의 **어투·스타일** 학습 |
| `data/rag_data/` | **RAG DB** | 이형 강의 기반 **지식·논리 근거** 확보 |

Few-Shot만 쓰면 말투는 비슷한데 내용이 얕고, RAG만 쓰면 내용은 맞는데 톤이 밋밋합니다.
**둘을 같이 써야** "이형처럼 말하면서 이형처럼 근거 있는" 피드백이 나옵니다.

---

## 📈 튜닝 진행 과정

| 단계 | 적용 | 결과 |
|---|---|---|
| 1 | Few-Shot only | 어투는 맞으나 내용이 피상적 |
| 2 | **+ RAG** | 논리·근거 강화 |
| 3 | **+ 프롬프트 엔지니어링** | 이형 스타일 규칙 9종 적용 |
| 4 | **+ 타임스탬프 / 비언어 규칙** | 영상 분석 대응 (최종) |

---

## 🏗️ 시스템 아키텍처

```
   면접 영상 (.mp4)  또는  답변 텍스트
              │
              ▼
   ┌──────────────────────┐
   │  Gemini File API     │  영상 업로드 → 음성·표정·자세 직접 인식
   └──────────┬───────────┘  (별도 STT 불필요)
              │
              ▼  키워드 추출
   ┌──────────────────────┐
   │      ChromaDB        │  paraphrase-multilingual-MiniLM-L12-v2
   │   786 chunks (RAG)   │  → 관련 청크 Top-4 검색
   └──────────┬───────────┘
              │
              ▼
   ┌────────────────────────────────────┐
   │   Few-Shot (Q./A. 형식) 5쌍         │  ← feedback_data
   │   + RAG 참고자료 4청크               │  ← rag_data
   │   + 영상 파일                        │
   │   + 시스템 프롬프트 (이형 스타일 규칙) │
   └──────────────┬─────────────────────┘
                  │
                  ▼
        gemini-3.1-pro-preview
                  │
                  ▼
    [MM:SS] 타임스탬프 기반 이형 스타일 피드백
```

---

## 🗂️ 프로젝트 구조

```
interview-coach/
│
├── 01_rag_build.ipynb          RAG 지식베이스 구축 → chroma_db/ (786청크)
├── 01b_chunk_experiment.py     청킹 파라미터 비교 실험 (200/300/500)
├── 02_eval_pipeline.ipynb      자동 평가 (LLM-as-judge, 6개 메트릭)
├── 03_feedback.ipynb           초기 버전 — RAG only
├── 04_fewshot_feedback.ipynb   ⭐ RAG + Few-Shot (텍스트 입력)
├── 05_video_feedback.ipynb     ⭐ RAG + Few-Shot + 영상 (멀티모달, 최종)
├── gemini_튜닝.ipynb            Gemini 파인튜닝 실험
│
├── data/
│   ├── rag_data/               RAG DB용 — 이형 강의 정리 (51개, 5카테고리)
│   ├── feedback_data/          Few-Shot용 — 답변·피드백 쌍 (97개)
│   └── finetune/               파인튜닝 데이터셋 (jsonl)
│
├── data_prep/
│   └── LLM용JSON.ipynb          파인튜닝 데이터 생성
│
└── models/
    ├── llm_comparison/         LLM 선정 비교 실험
    │   ├── openAI_튜닝.ipynb
    │   └── vertex_ai.ipynb
    └── nonverbal/              비언어 분석 실험 (7종)
        ├── whisper_test.ipynb  crepe_test.ipynb  silero_test.ipynb
        ├── DeepFace.ipynb      표정분석.ipynb
        └── 고개_각도_측정.ipynb  몸_움직임_측정.ipynb
```

---

## 📦 데이터셋

### RAG 지식베이스 — `data/rag_data/` (51개)
| 카테고리 | 파일 수 |
|---|---|
| 1. 핵심 역량(필살기) 구축 및 취업 전략 | 12 |
| 2. 비즈니스 커뮤니케이션 및 답변 스킬 | 8 |
| 3. 면접 단계-유형별 맞춤 공략 | 12 |
| 4. 필수-압박 질문 완벽 방어 | 10 |
| 5. 면접 마인드셋 및 기본 태도 | 9 |

### Few-Shot 데이터 — `data/feedback_data/` (97개)
실제 면접 인터뷰 텍스트와 그에 대한 이형 피드백이 **1:1로 매칭**된 쌍 데이터.
`1~5` / `6~10` / `11~16` / `17~20` 그룹으로 구성.

---

## 🛠️ 기술 스택

| 구성 요소 | 기술 |
|---|---|
| **LLM** | `gemini-3.1-pro-preview` (멀티모달·영상 입력) |
| **벡터 DB** | ChromaDB 0.6.3 (로컬 영구 저장) |
| **임베딩** | paraphrase-multilingual-MiniLM-L12-v2 (384차원) |
| **청킹** | 300자 / overlap 50자, 문장 경계 분할 |
| **영상 처리** | Gemini File API (STT 별도 불필요) |
| **평가** | LLM-as-judge |
| **비언어 실험** | Whisper / CREPE / Silero / DeepFace / MediaPipe |

---

## 🚀 빠른 시작

### 1. 설치
```bash
git clone https://github.com/whdyddls4276/interview_coach.git
cd interview_coach
pip install -r requirements.txt
cp .env.example .env      # .env 에 GOOGLE_API_KEY 입력
```

### 2. RAG 지식베이스 구축 (최초 1회)
```
01_rag_build.ipynb 실행  →  chroma_db/ 생성 (786청크)
```

### 3. 피드백 받기

**텍스트 입력**
```
04_fewshot_feedback.ipynb  →  STEP 5 에서 ANSWER 수정 후 실행
```

**영상 입력 (최종 파이프라인)**
```
05_video_feedback.ipynb  →  STEP 5 에서 VIDEO_PATH 수정 후 실행
```

---

## ⚠️ 모델 쿼터 안내

`gemini-3.1-pro-preview` 는 **무료 등급 쿼터가 0**이라 결제가 활성화된 프로젝트에서만 호출됩니다.
무료로 테스트하려면 노트북 STEP 1 에서 폴백 모델로 교체하세요:

```python
MODEL = 'gemini-3.1-pro-preview'
# MODEL = 'gemini-2.5-flash'   # FALLBACK: 무료 등급 사용 가능
```

사용 가능한 모델 목록은 아래로 직접 확인할 수 있습니다:
```python
for m in client.models.list():
    print(m.name)
```

---

## 🔐 API 키 설정

`.env` 파일을 만들고 아래와 같이 입력하세요:
```
GOOGLE_API_KEY=your_google_api_key_here
```
> API 키는 절대 코드에 직접 넣지 마세요. `.gitignore` 에 `.env` 가 포함되어 있습니다.

---

## 🔬 LLM 비교 실험

`models/llm_comparison/` 에서 모델 선정을 위한 비교를 진행했습니다.

| 모델 | 방식 | 결과 |
|---|---|---|
| Gemini 3.1 Pro | RAG + Few-Shot + 멀티모달 | ✅ **최종 채택** — 영상 직접 분석 가능 |
| GPT-4.1 mini | 파인튜닝 | 텍스트 한정, 영상 불가 |
| Gemini Flash Lite | RAG + 프롬프트 | 무료지만 피드백 깊이 부족 |

---

## 📊 평가 메트릭

`02_eval_pipeline.ipynb` 에서 측정하는 항목입니다.

| 메트릭 | 설명 | 범위 |
|---|---|---|
| `factpunch` | 팩폭 한줄평의 날카로움 | 1–5 |
| `analysis` | 시선 분석의 깊이 | 1–5 |
| `prescription` | 처방전의 실천 가능성 | 1–5 |
| `accuracy` | RAG 근거 활용도 | 1–5 |
| `context_relevance` | 검색 청크 관련성 | 0–1 |
| `keyword_hit_rate` | 핵심 키워드 포함율 | 0–1 |

---

## 📌 향후 계획

- [ ] `models/nonverbal/` 실험 모듈을 메인 파이프라인에 정식 통합
- [ ] 웹 UI (Streamlit) 연동
- [ ] RAG 데이터 확장 (현재 51개 → 목표 200개+)
- [ ] 다중 질문 연속 면접 시뮬레이션
