"""
- 모델: SentenceTransformer("jhgan/ko-sroberta-multitask")
- 입력:
    function_query_context_sample.csv  -> 질의 목록
    process_embedding_contexts_sample.csv -> 컨텍스트 목록
- 출력:
    semantic_search_results.csv
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# ---------- 설정 ----------
QUERY_CSV = "./function_query_context.csv"
CONTEXT_CSV = "./process_embedding_contexts.csv"
OUTPUT_CSV = "./semantic_search_results_old.csv"

# set pandas display options to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def load_queries(path: str):
    """query CSV 전처리, 컬럼 추출"""
    df = pd.read_csv(path)

    if not {'system', '업무기능', 'context'}.issubset(df.columns.astype(str).str.lower()):
        raise ValueError("'context' 컬럼이 필요합니다.")

    # 1. 문자열 정리
    # query 가 "" 이면 에러가 발생하므로 주의
    df['CONTEXT'] = df['CONTEXT'].fillna("NaN").astype(str).str.strip()

    # 2. 문장 내부에 존재하는 'N/A', 'NaN', 'None', 'null' 등 제거
    remove_patterns = [
        r"\bN/?A\b",
        r"\bNaN\b",
        r"\bNONE\b",
        r"\bNULL\b"
    ]
    pattern = re.compile("|".join(remove_patterns), flags=re.IGNORECASE)
    df['CONTEXT'] = df['CONTEXT'].astype(str).str.replace(pattern, " ", regex=True)

    # 2.1 한개 이상의 공백을 1개의 공백으로 대치
    df['CONTEXT'] = df['CONTEXT'].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    
    return df

def load_contexts(path: str):
    """context CSV code, context 컬럼 전처리, 추출"""    
    df = pd.read_csv(path)

    if not {'code', 'contexts'}.issubset(df.columns.astype(str).str.lower()):
        raise ValueError("'contexts' 컬럼이 필요합니다.")    

    df['CONTEXTS'] = df['CONTEXTS'].fillna("NaN").astype(str).str.strip()

    # 2. 문장 내부에 존재하는 'N/A', 'NaN', 'None', 'null' 등 제거
    remove_patterns = [
        r"\bN/?A\b",
        r"\bNaN\b",
        r"\bNONE\b",
        r"\bNULL\b"
    ]
    pattern = re.compile("|".join(remove_patterns), flags=re.IGNORECASE)
    df['CONTEXTS'] = df['CONTEXTS'].astype(str).str.replace(pattern, " ", regex=True)

    # 2.1 한개 이상의 공백을 1개의 공백으로 대치
    df['CONTEXTS'] = df['CONTEXTS'].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    return df

def main():
    # 1) 데이터 로드
    queries_df = load_queries(QUERY_CSV)
    queries = queries_df['CONTEXT'].tolist()
    contexts_df = load_contexts(CONTEXT_CSV)
    contexts_text = contexts_df['CONTEXTS'].tolist()    
    pr_nm_df = contexts_df[['Pr_Lv1_Nm','Pr_Lv2_Nm','Pr_Lv3_Nm','Pr_Lv4_Nm','Pr_Lv5_Nm']]
    # 값이 있는 맨 마지막 컬럼의 값 추출
    pr_nm = pr_nm_df.stack().groupby(level=0).last().reindex(pr_nm_df.index)

    # 2) 모델 로드
    model_path = './ko-sroberta'    # for offline use
    model = SentenceTransformer(model_path)

    # 3) 임베딩 계산
    q_emb = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True)
    c_emb = model.encode(contexts_text, convert_to_tensor=True, normalize_embeddings=True)

    # 4) 유사도 계산
    sim = util.cos_sim(q_emb, c_emb)

    # 5) 각 query별 상위 5개 추출
    topk = 5
    k = min(topk, len(contexts_text))
    scores, idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=True)

    # 6) 결과 DataFrame 생성
    result_rows = []
    for qi, query in enumerate(queries):
        for rank in range(k):
            ci = idx[qi, rank].item()
            result_rows.append({
                "system": queries_df.iloc[qi]["SYSTEM"],
                "menu_lv1_nm": queries_df.iloc[qi]["MENU_LV1_NM"],
                "menu_lv2_nm": queries_df.iloc[qi]["MENU_LV2_NM"],
                "menu_lv3_nm": queries_df.iloc[qi]["MENU_Lv3_NM"],
                "menu_lv4_nm": queries_df.iloc[qi]["MENU_Lv4_NM"],
                "menu_lv5_nm": queries_df.iloc[qi]["MENU_Lv5_NM"],
                "function": queries_df.iloc[qi]["업무기능"],
                "query": query,
                "rank": rank+1,
                "similarity_score": round(float(scores[qi, rank].item()), 6),
                "code": contexts_df.iloc[ci]["CODE"],
                "pr_nm": pr_nm.iloc[ci],
                "pr_context": contexts_df.iloc[ci]["CONTEXTS"]
            })

    result_df = pd.DataFrame(result_rows, columns=[
        "system",
        "menu_lv1_nm",
        "menu_lv2_nm",
        "menu_lv3_nm",
        "menu_lv4_nm",
        "menu_lv5_nm",
        "function",
        "query",
        "rank",
        "similarity_score",
        "code",
        "pr_nm",
        "pr_context"
        ])
    
    # 7) CSV 저장
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"총 {len(queries)}개의 query 처리 완료.")
    print(f"저장 경로: {OUTPUT_CSV}")
    print(result_df.head(10))

if __name__ == "__main__":
    main()