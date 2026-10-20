"""
eval_v2.py — v1(04·05 노트북) vs v2(rag_v2 + prompts_v2) 평가

1) 검색: 질문 14개, 정답 = 그 주제를 제목으로 다루는 강의. 유사도 점수가 아니라 정답 강의를 찾았는지로 잰다
   - v1      : 기존 DB(interview_rag, 모의면접 전사 포함) + 임베딩 검색
   - v2-임베딩: 강의만 DB + 임베딩 검색       (데이터를 바꾼 효과)
   - v2      : 강의만 DB + 하이브리드 검색    (검색 방식까지 바꾼 효과)
2) 생성: data/feedback_data 중 6쌍을 평가용으로 빼고(예시 풀에서 같은 영상 제외), 같은 모델·같은 예시로
   v1(기존 프롬프트·기존 검색) vs v2(새 프롬프트·새 검색) 비교
   - 형식: 제목·글머리표 줄, 굵은 글씨, 구어체 끝맺음 비율, 길이
   - 평가 방향: 실제 코치 피드백과 모델 피드백을 각각 긍정/혼합/부정으로 판정(LLM 판정)해 일치하는지

실행: python eval_v2.py            (결과: eval_results_v2/)
      python eval_v2.py --retrieval (검색만, API 호출 없음)
"""
import json, os, random, re, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
BASE   = Path(__file__).parent
PROJ   = BASE.parent / '프로젝트3(면접)'
OUT    = BASE / 'eval_results_v2'
OUT.mkdir(exist_ok=True)
GEN_MODEL   = 'gemini-3.5-flash'           # 3.1 Pro는 무료 등급 한도 0 → 두 버전 모두 같은 모델로 비교
JUDGE_MODEL = 'gemini-flash-lite-latest'

QUESTIONS = {  # 질문: 정답 강의 제목에 들어 있는 말
    '1분 자기소개는 어떻게 준비해야 하나요': ['1분 자기소개', '1분자기소개'],
    '지원동기는 어떻게 말해야 하나요': ['지원동기'],
    '퇴직 사유를 물어보면 어떻게 답해야 하나요': ['퇴직사유'],
    '면접에서 두괄식으로 답변하는 방법이 궁금해요': ['두괄식'],
    '압박 질문에는 어떻게 대처하나요': ['압박'],
    '면접에서 쓰면 안 되는 표현이 있나요': ['표현'],
    '나이가 많은 지원자는 면접에서 불리한가요': ['나이 많은'],
    '떨어지면 어디 갈 거냐는 질문에는 어떻게 답하나요': ['떨어지면 어디'],
    '면접에서 긴장하거나 떨리면 어떻게 하나요': ['말 더듬', '불안증', '떨리는', '떨지 않는'],
    '임원 면접은 어떻게 준비하나요': ['임원면접'],
    '입사 후 포부는 어떻게 말해야 하나요': ['입사 후 포부'],
    '가장 힘들었던 경험은 어떻게 답하나요': ['가장 힘들었던'],
    'PT 면접은 어떻게 준비하나요': ['PT 면접'],
    '면접 복장은 어떻게 입어야 하나요': ['복장'],
}
rel = lambda q, src: any(k in src for k in QUESTIONS[q])


def score_ranking(name, sources_by_q, corpus_sources):
    ans = [q for q in QUESTIONS if any(rel(q, s) for s in corpus_sources)]
    h1 = h3 = mrr = 0
    for q in ans:
        ranks = [r for r, s in enumerate(sources_by_q[q][:5], 1) if rel(q, s)]
        h1 += bool(ranks and ranks[0] == 1); h3 += bool(ranks and ranks[0] <= 3)
        mrr += 1 / ranks[0] if ranks else 0
    top1 = [sources_by_q[q][0] for q in QUESTIONS]
    return {'name': name, 'answerable': len(ans), 'answerable_q': ans, 'hit1': h1, 'hit3': h3,
            'mrr5': round(mrr / max(1, len(ans)), 3),
            'same_top1_max': max(top1.count(s) for s in top1),
            'top1': {q: sources_by_q[q][0][:50] for q in QUESTIONS}}


def eval_retrieval():
    import chromadb
    from chromadb.utils import embedding_functions
    from rag_v2 import CHROMA_DIR, COLLECTION, EMBED_MODEL, HybridRetriever
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    res = []

    # v1: 기존 DB는 복사본을 열어 원본을 건드리지 않는다
    tmp = Path(tempfile.mkdtemp()) / 'chroma_db_v1'
    shutil.copytree(PROJ / 'chroma_db', tmp)
    v1 = chromadb.PersistentClient(path=str(tmp)).get_collection('interview_rag', embedding_function=ef)
    v1_src = {m['source'] for m in v1.get(include=['metadatas'], limit=100000)['metadatas']}
    by_q = {q: [m['source'] for m in v1.query(query_texts=[q], n_results=5)['metadatas'][0]] for q in QUESTIONS}
    res.append(score_ranking('v1 기존 DB + 임베딩', by_q, v1_src))

    v2 = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION, embedding_function=ef)
    v2_src = {m['source'] for m in v2.get(include=['metadatas'], limit=100000)['metadatas']}
    by_q = {q: [m['source'] for m in v2.query(query_texts=[q], n_results=5)['metadatas'][0]] for q in QUESTIONS}
    res.append(score_ranking('v2 강의만 DB + 임베딩', by_q, v2_src))

    rag = HybridRetriever()
    by_q = {q: [h['source'] for h in rag.search(q, 5)] for q in QUESTIONS}
    res.append(score_ranking('v2 강의만 DB + 하이브리드', by_q, v2_src))

    common = [q for q in res[0]['answerable_q'] if q in res[2]['answerable_q']]
    print('\n[검색] 정답 강의를 찾았나 (상위 5개 기준)')
    for r in res:
        print(f"  {r['name']:22s} 정답 있는 질문 {r['answerable']:2d}개 | Hit@1 {r['hit1']:2d} | Hit@3 {r['hit3']:2d} | "
              f"MRR@5 {r['mrr5']:.2f} | 같은 1위 최대 {r['same_top1_max']}회")
    json.dump({'results': res, 'common_questions': common}, open(OUT / 'retrieval.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    return res


# ── 생성 평가 ─────────────────────────────────────────────────
def load_pairs():
    FB = BASE / 'data' / 'feedback_data'
    pairs = []
    for grp, ivd, fbd in (('1~5', 'Interview_text', 'Feedback_text'), ('11~16', 'interview_text', 'feedback_text')):
        for iv in sorted((FB / grp / ivd).glob('*.txt')):
            fb = FB / grp / fbd / iv.name
            if fb.exists():
                pairs.append({'id': f'{grp}/{iv.stem}', 'video': f'{grp}/{iv.stem.split("_")[0]}',
                              'interview': iv.read_text(encoding='utf-8').strip(),
                              'feedback': fb.read_text(encoding='utf-8').strip()})
    td = FB / '6~10' / 'text'
    for iv in sorted(td.glob('*인터뷰*.txt')):
        n = re.match(r'(\d+)', iv.name)
        if not n:
            continue
        fb = '\n\n'.join(f.read_text(encoding='utf-8').strip() for f in sorted(td.glob(f'{n.group(1)}_[0-9]*.txt')))
        pairs.append({'id': f'6~10/{n.group(1)}', 'video': f'6~10/{n.group(1)}',
                      'interview': iv.read_text(encoding='utf-8').strip(), 'feedback': fb})
    return [p for p in pairs if p['interview'] and p['feedback']]


def form_metrics(t):
    lines = [l for l in t.split('\n') if l.strip()]
    sents = [s for s in re.split(r'(?<=[.?!])\s+|\n+', t) if len(re.sub(r'[\s*#>\-]', '', s)) > 3]
    col = np.mean([bool(re.search(r'(요|죠|거든요|잖아요|네요|어요|아요|세요)[.!?”"*)\s]*$', s.strip())) for s in sents]) if sents else 0
    return {'heading_or_bullet_lines': sum(bool(re.match(r'\s*(#|[-*•]\s|\d+\.\s)', l)) for l in lines),
            'bold': t.count('**'), 'colloquial_end': round(float(col), 2), 'chars': len(t)}


def main():
    eval_retrieval()
    if '--retrieval' in sys.argv:
        return
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    import chromadb
    from chromadb.utils import embedding_functions
    from rag_v2 import EMBED_MODEL, HybridRetriever
    from prompts_v2 import SYSTEM_PROMPT_V1, system_prompt

    load_dotenv(PROJ / '.env')
    client = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY', ''))
    pairs = load_pairs()
    test = pairs[::5][:6]
    test_videos = {p['video'] for p in test}
    pool = [p for p in pairs if p['video'] not in test_videos]
    shots = random.Random(0).sample(pool, 5)           # 두 버전에 같은 예시

    tmp = Path(tempfile.mkdtemp()) / 'chroma_db_v1'
    shutil.copytree(PROJ / 'chroma_db', tmp)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    v1 = chromadb.PersistentClient(path=str(tmp)).get_collection('interview_rag', embedding_function=ef)
    rag = HybridRetriever()

    def call(model, system, contents, max_tokens=4096):
        for k in range(6):
            try:
                r = client.models.generate_content(model=model, contents=contents, config=types.GenerateContentConfig(
                    system_instruction=system, temperature=0.4, max_output_tokens=max_tokens))
                time.sleep(6)
                if r.text:
                    return r.text.strip()
            except Exception as e:
                print('   재시도:', str(e)[:60]); time.sleep(30)
        return ''

    def contents_for(answer, ctx):
        c = []
        for s in shots:
            c += [types.Content(role='user', parts=[types.Part.from_text(text=f'Q.\n{s["interview"]}')]),
                  types.Content(role='model', parts=[types.Part.from_text(text=f'A.\n{s["feedback"]}')])]
        c.append(types.Content(role='user', parts=[types.Part.from_text(
            text=f'Q.\n{answer}\n\n--- 참고 자료 (면접왕 이형 강의) ---\n{ctx}')]))
        return c

    path = OUT / 'generation_outputs.json'
    gen = json.load(open(path, encoding='utf-8')) if path.exists() else {}
    for p in test:
        g = gen.setdefault(p['id'], {'interview': p['interview'], 'coach': p['feedback']})
        if not g.get('v1'):
            docs = v1.query(query_texts=[p['interview'][:200]], n_results=4)['documents'][0]
            g['v1'] = call(GEN_MODEL, SYSTEM_PROMPT_V1, contents_for(p['interview'], '\n\n'.join(f'[참고{i+1}] {d}' for i, d in enumerate(docs))))
        if not g.get('v2'):
            hits = rag.search(p['interview'][:300], 4)
            g['v2_sources'] = [h['source'] for h in hits]
            g['v2'] = call(GEN_MODEL, system_prompt(), contents_for(p['interview'], '\n\n'.join(f'[참고{i+1}] {h["text"]}' for i, h in enumerate(hits))))
        for who in ('coach', 'v1', 'v2'):
            key = f'judge_{who}'
            if g.get(who) and not g.get(key):
                lab = call(JUDGE_MODEL, '너는 평가자다. 한 단어로만 답한다.', [
                    f'[면접 답변]\n{p["interview"][:1500]}\n\n[피드백]\n{g[who][:2500]}\n\n'
                    '이 피드백은 답변을 전체적으로 어떻게 평가하는가? 긍정 / 혼합 / 부정 중 하나로만 답하라.'], 20)
                g[key] = next((w for w in ('긍정', '혼합', '부정') if w in lab), lab)
        print(f"  {p['id']:14s} v1 {len(g.get('v1', ''))}자 · v2 {len(g.get('v2', ''))}자 | 판정 코치 {g.get('judge_coach')} · v1 {g.get('judge_v1')} · v2 {g.get('judge_v2')}")
        json.dump(gen, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    done = [g for g in gen.values() if g.get('v1') and g.get('v2')]
    summ = {}
    for who in ('coach', 'v1', 'v2'):
        ms = [form_metrics(g[who]) for g in done]
        summ[who] = {k: round(float(np.mean([m[k] for m in ms])), 2) for k in ms[0]}
        if who != 'coach':
            pairs_j = [(g.get(f'judge_{who}'), g.get('judge_coach')) for g in done if g.get(f'judge_{who}') and g.get('judge_coach')]
            summ[who]['direction_match'] = f"{sum(a == b for a, b in pairs_j)}/{len(pairs_j)}"
    print(f'\n[생성] 평가 {len(done)}건 (같은 모델 {GEN_MODEL}, 같은 예시 5개)')
    print(f"  {'':6s} 제목·글머리표 줄  굵은글씨  구어체 끝맺음  길이   평가 방향 일치(코치 기준)")
    for who in ('coach', 'v1', 'v2'):
        s = summ[who]
        print(f"  {who:6s} {s['heading_or_bullet_lines']:10.1f} {s['bold']:9.1f} {s['colloquial_end']:12.2f} {s['chars']:7.0f}   {s.get('direction_match', '-')}")
    json.dump(summ, open(OUT / 'generation_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
