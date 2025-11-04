"""
한국어 텍스트를 위한 SBERT와 FAISS 기반 시맨틱 검색

이 스크립트는 SentenceTransformers('jhgan/ko-sroberta-multitask')를 사용해 한국어 텍스트를 고품질 임베딩으로 변환하고,
FAISS를 사용해 효율적인 벡터 인덱싱 및 검색을 수행합니다. 쿼리 CSV와 컨텍스트 CSV를 입력으로 받아 각 쿼리에 대해 가장 유사한
상위 k개의 컨텍스트를 찾아 CSV 파일로 결과를 저장합니다. FAISS로 후보를 선별한 후 SBERT로 정밀한 코사인 유사도를 계산하는
하이브리드 검색 방식을 사용해 정확도와 속도를 모두 최적화합니다. 추가로, 유사도 값 분포를 막대그래프로 시각화하여 저장합니다.

주요 기능:
- 'jhgan/ko-sroberta-multitask' 모델로 한국어 임베딩 생성 (KLUE-STS 점수 ~85%).
- FAISS 'IndexIVFFlat'으로 높은 정확도의 근사 최근접 이웃(ANN) 검색.
- 하이브리드 검색: FAISS로 상위 10개 후보 선별 후 SBERT로 정확한 코사인 유사도 계산.
- 데이터 크기에 따라 동적으로 'nprobe' 조정해 정확도와 속도 균형 유지.
- 유사도 임계값(0.5)을 설정해 낮은 신뢰도의 결과 필터링.
- 유사도 값 분포를 구간별(0.7 이상, 0.6~0.7, 0.5~0.6, 0.4~0.5, 0.4 이하) 막대그래프로 시각화.

입력:
- 'function_query_context.csv': 쿼리 csv 파일 (컬럼: ['system', '업무기능', 'context', 'menu_lv1_nm', ...]).
- 'process_embedding_contexts.csv': 컨텍스트 csv 파일 (컬럼: ['code', 'contexts', 'pr_lv1_nm', ...]).

출력:
- 'semantic_search_results.csv': 쿼리 메타데이터, 상위 5개 매칭 결과, 유사도 점수, 컨텍스트 정보 포함.
- 'similarity_distribution.png': 유사도 값 분포를 나타내는 막대그래프.

의존성:
- sentence-transformers
- pandas
- numpy
- torch
- faiss-cpu (GPU 지원 시 faiss-gpu)
- matplotlib (유사도 분포 시각화용)

사용법:
    # 입력 CSV 파일과 모델 경로('./ko-sroberta')가 준비되어 있어야 합니다.
    # 결과는 './semantic_search_results.csv'와 './similarity_distribution.png'에 저장됩니다.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import re
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import faiss
import matplotlib.pyplot as plt


# 파일 세트 목록 (각 세트별로 입력/출력/인덱스/그래프 파일 경로를 통합)
FILE_TASKS = [
    # {
    #     "query_csv": "./data/function_건축_.csv",
    #     "context_csv": "./data/process_건축_.csv",
    #     "output_csv": "./result/semantic_search_results_건축.csv",
    #     "index_file": "./result/faiss_index1.ivfflat",
    #     "plot_file": "./result/similarity_distribution_건축.png"
    # },
    # {
    #     "query_csv": "./data/function_토목_.csv",
    #     "context_csv": "./data/process_토목_.csv",
    #     "output_csv": "./result/semantic_search_results_토목.csv",
    #     "index_file": "./result/faiss_index2.ivfflat",
    #     "plot_file": "./result/similarity_distribution_토목.png"
    # },
    # {
    #     "query_csv": "./data/function_플랜트_.csv",
    #     "context_csv": "./data/process_플랜트_.csv",
    #     "output_csv": "./result/semantic_search_results_플랜트.csv",
    #     "index_file": "./result/faiss_index3.ivfflat",
    #     "plot_file": "./result/similarity_distribution_플랜트.png"
    # },
    # {
    #     "query_csv": "./data/function_주택_.csv",
    #     "context_csv": "./data/process_주택_.csv",
    #     "output_csv": "./result/semantic_search_results_주택.csv",
    #     "index_file": "./result/faiss_index4.ivfflat",
    #     "plot_file": "./result/similarity_distribution_주택.png"
    # },
    {
        "query_csv": "./data/function_하이테크_.csv",
        "context_csv": "./data/process_하이테크_.csv",
        "output_csv": "./result_test/semantic_search_results_하이테크.csv",
        "index_file": "./result_test/faiss_index5.ivfflat",
        "plot_file": "./result_test/similarity_distribution_하이테크.png"
    },
    {
        "query_csv": "./data/test_query.csv",
        "context_csv": "./data/test_context.csv",
        "output_csv": "./result_test/test_result.csv",
        "index_file": "./result_test/test.ivfflat",
        "plot_file": "./result_test/test_similarity.png"
    },
]
BATCH_SIZE = 128
SIMILARITY_THRESHOLD = 0.5
TOPK_CANDIDATES = 10
TOPK_OUTPUT = 5
NLIST_MIN = 200
NPROBE_MIN = NLIST_MIN // 5

# pandas 출력 설정
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def load_queries(path: str):
    df = pd.read_csv(path)
    if not {'system', '업무기능', 'context'}.issubset(df.columns.astype(str).str.lower()):
        raise ValueError("'context' 컬럼이 필요합니다.")
    return df

def load_contexts(path: str):
    df = pd.read_csv(path)
    if not {'code', 'contexts'}.issubset(df.columns.astype(str).str.lower()):
        raise ValueError("'contexts' 컬럼이 필요합니다.")
    return df

def build_faiss_index(model, contexts_text, index_file=None):
    """
    컨텍스트 임베딩을 위한 FAISS 인덱스를 생성하고 저장합니다.

    'IndexIVFFlat'을 사용해 벡터 압축 없이 높은 정확도를 유지합니다. GPU 가속을 지원하며,
    인덱스와 임베딩을 파일로 저장해 재사용 가능합니다.

    Parameters
    ----------
    model : SentenceTransformer
        임베딩 생성을 위한 사전 학습된 SentenceTransformer 모델.
    contexts_text : list of str
        인덱싱할 컨텍스트 텍스트 리스트.
    index_file : str, optional
        FAISS 인덱스 저장 경로 (기본값: './faiss_index.ivfflat').

    Returns
    -------
    faiss.Index
        학습된 FAISS 인덱스.
    numpy.ndarray
        컨텍스트 임베딩 배열 (N x 768).
    """
    embeddings = []
    for i in range(0, len(contexts_text), BATCH_SIZE):
        batch = contexts_text[i:i + BATCH_SIZE]
        batch_emb = model.encode(batch, convert_to_tensor=True, normalize_embeddings=True).cpu().numpy()
        embeddings.append(batch_emb)

    corpus_embeddings = np.concatenate(embeddings, axis=0).astype('float32')
    d = corpus_embeddings.shape[1]
    nlist = max(NLIST_MIN, min(2000, int(2 * np.sqrt(len(corpus_embeddings)))))

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.train(corpus_embeddings)
    index.add(corpus_embeddings)

    if faiss.get_num_gpus() > 0:
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, index_file)
    emb_file = index_file.replace('.ivfflat', '_embeddings.npy')
    np.save(emb_file, corpus_embeddings)

    print(f"FAISS 인덱스 빌드 완료. 저장 경로: {index_file}")
    print(f"FAISS 임베딩 저장 경로: {emb_file}")

    return index, corpus_embeddings

def load_faiss_index(index_file=None):
    """
    저장된 FAISS 인덱스와 임베딩을 함께 로드합니다.

    Parameters
    ----------
    index_file : str, optional
        FAISS 인덱스 파일 경로 (기본값: './faiss_index.ivfflat').

    Returns
    -------
    faiss.Index
        로드된 FAISS 인덱스 (GPU 가능 시 GPU로 전환).
    numpy.ndarray
        저장된 임베딩 배열 (N x 768)

    Raises
    ------
    FileNotFoundError
        인덱스 파일이 존재하지 않는 경우.
    """
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"{index_file} not found.")

    index = faiss.read_index(index_file)
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    emb_file = index_file.replace('.ivfflat', '_embeddings.npy')
    corpus_embeddings = np.load(emb_file)

    print(f"저장된 FAISS 인덱스 로드 완료 : {index_file}")
    print(f"저장된 FAISS 임베딩 로드 완료 : {emb_file}")

    return index, corpus_embeddings

def plot_similarity_distribution(scores, output_file=None):
    """
    유사도 값 분포를 막대그래프로 시각화하고 저장합니다.

    유사도 값을 구간(0.7 이상, 0.6~0.7, 0.5~0.6, 0.4~0.5, 0.4 이하)으로 나누어
    빈도를 계산하고, 한국어 폰트를 사용해 막대그래프를 생성합니다.

    Parameters
    ----------
    scores : list of float
        계산된 유사도 값 리스트.
    output_file : str, optional
        그래프 저장 경로 (기본값: './similarity_distribution.png').
    """
    # 구간 정의
    bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    labels = ['0.4 이하', '0.4~0.5', '0.5~0.6', '0.6~0.7', '0.7~0.8', '0.8 이상']
    hist, _ = np.histogram(scores, bins=bins)

    # 한국어 폰트 설정 (Windows: Malgun Gothic, macOS/Linux: Noto Sans CJK)
    plt.rcParams['font.family'] = 'Malgun Gothic' if os.name == 'nt' else 'Noto Sans CJK KR'
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

    # 막대그래프 생성
    plt.figure(figsize=(5, 5))
    bar = plt.bar(labels, hist, color='skyblue', edgecolor='black')
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.1f' % height, ha='center', va='bottom', size = 9)

    plt.title('유사도 값 분포', fontsize=12, pad=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # 그래프 저장
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"유사도 분포 그래프 저장 완료: {output_file}")


# 여러 파일 세트 반복 처리 함수
def semantic_search(file_tasks):
    """
    쿼리와 컨텍스트 간 시맨틱 검색을 수행하고 결과를 저장합니다.

    단계:
    1. 쿼리 및 컨텍스트 CSV 로드 및 전처리.
    2. SentenceTransformer 모델 로드 ('jhgan/ko-sroberta-multitask').
    3. FAISS 인덱스 생성 또는 로드.
    4. 쿼리 임베딩을 배치 단위로 생성.
    5. 하이브리드 검색: FAISS로 상위 10개 후보 선별, SBERT로 코사인 유사도 재계산.
    6. 유사도 임계값(0.5)으로 결과 필터링.
    7. 쿼리 메타데이터, 상위 5개 매칭, 유사도 점수를 CSV로 저장.
    8. 유사도 값 분포를 막대그래프로 시각화하여 저장.

    참고:
    - 하이브리드 검색을 위해 'corpus_embeddings.npy' 파일이 필요합니다.
    - 동적 'nprobe' (min(200, nlist//5))로 정확도와 속도 균형 유지.
    - GPU 가속 지원.
    - 유사도 분포는 'similarity_distribution.png'에 저장.
    """
    # --- PEFT(LoRA) 모델 로드 방식으로 변경 ---
    # 1. 원본 기반 모델 경로
    base_model_path = './ko-sroberta' # 'jhgan/ko-sroberta-multitask'의 로컬 경로
    # 2. 학습된 PEFT 어댑터 경로
    peft_path = './output/ko-sroberta-expert-lora'

    print(f"--- 모델 로드 시작 ---")
    print(f"    Base model: {base_model_path}")
    print(f"    PEFT adapter: {peft_path}")

    # SentenceTransformer 모델을 로드, PEFT 어댑터 적용
    model = SentenceTransformer(base_model_path)
    print(f"--- Base 모델 로드 완료  ---")
    if peft_path:
        model.load_adapter(peft_path)
        print(f"--- PEFT 모델 로드 완료 ---")

    for i, task in enumerate(file_tasks):
        required_keys = ["query_csv", "context_csv", "output_csv", "index_file", "plot_file"]
        for k in required_keys:
            if k not in task:
                raise ValueError(f"{i+1}번째 파일 세트에 '{k}' 키가 없습니다.")

        QUERY_CSV = task["query_csv"]
        CONTEXT_CSV = task["context_csv"]
        OUTPUT_CSV = task["output_csv"]
        INDEX_FILE = task["index_file"]
        PLOT_FILE = task["plot_file"]

        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(PLOT_FILE), exist_ok=True)

        print(f"\n==== {i+1}번째 파일셋 처리 ====")
        queries_df = load_queries(QUERY_CSV)
        queries = queries_df['context'].tolist()
        contexts_df = load_contexts(CONTEXT_CSV)
        contexts_text = contexts_df['contexts'].tolist()
        pr_nm_df = contexts_df[['pr_lv1_nm','pr_lv2_nm','pr_lv3_nm','pr_lv4_nm','pr_lv5_nm']]
        pr_nm = pr_nm_df.stack().groupby(level=0).last().reindex(pr_nm_df.index)    # ""이 아닌 마지막 셀값 가져오기.  blank cell " " is not working

        if not os.path.exists(INDEX_FILE):
            index, corpus_embeddings = build_faiss_index(model, contexts_text, index_file=INDEX_FILE)
        else:
            index, corpus_embeddings = load_faiss_index(index_file=INDEX_FILE)

        nlist = index.nlist
        index.nprobe = min(NPROBE_MIN, nlist // 5)
        print(f'INFO : nlist(number of cluster centroids) = {nlist}')
        print(f'INFO : nProbe(number of probing clusters) = {index.nprobe}')

        q_emb_list = []
        for j in range(0, len(queries), BATCH_SIZE):
            batch = queries[j:j + BATCH_SIZE]
            batch_emb = model.encode(batch, convert_to_tensor=True, normalize_embeddings=True).cpu().numpy().astype('float32')
            q_emb_list.append(batch_emb)
        q_emb = np.concatenate(q_emb_list, axis=0)

        result_rows = []
        similarity_scores = []
        corpus_emb_tensor = torch.from_numpy(corpus_embeddings).to(torch.float32)
        for qi in range(len(queries)):
            q_emb_tensor = torch.from_numpy(q_emb[qi:qi+1]).to(torch.float32)
            D, I = index.search(q_emb[qi:qi+1], k=TOPK_CANDIDATES)
            candidate_idx = I[0]
            candidate_emb = corpus_emb_tensor[candidate_idx]
            precise_scores = util.cos_sim(q_emb_tensor, candidate_emb)[0].numpy()
            sorted_indices = np.argsort(precise_scores)[::-1][:TOPK_OUTPUT]
            top_scores = precise_scores[sorted_indices]
            top_indices = candidate_idx[sorted_indices]
            for rank, (ci, score) in enumerate(zip(top_indices, top_scores)):
                if ci == -1 or score < SIMILARITY_THRESHOLD:
                    print(f'WARNING: 예측에 실패했거나 threshold(0.5) 이하입니다.  query no.= {qi}, ci= {ci}, score= {score}')
                    # continue      # 점수가 낮아도 출력
                similarity_scores.append(score)
                result_rows.append({
                    "system": queries_df.iloc[qi]["system"],
                    "menu_lv1_nm": queries_df.iloc[qi]["menu_lv1_nm"],
                    "menu_lv2_nm": queries_df.iloc[qi]["menu_lv2_nm"],
                    "menu_lv3_nm": queries_df.iloc[qi]["menu_lv3_nm"],
                    "menu_lv4_nm": queries_df.iloc[qi]["menu_lv4_nm"],
                    "menu_lv5_nm": queries_df.iloc[qi]["menu_lv5_nm"],
                    "function": queries_df.iloc[qi]["업무기능"],
                    "query": queries[qi],
                    "user_column_1": "",
                    "user_column_2": "",
                    "rank": rank+1,
                    "similarity_score": round(float(score), 6),
                    "code": contexts_df.iloc[ci]["code"],
                    "pr_nm": pr_nm.iloc[ci],
                    "pr_context": contexts_df.iloc[ci]["contexts"],
                    "menu_id": queries_df.iloc[qi]["menu_id"],
                    "program_id": queries_df.iloc[qi]["program_id"],
                })

        if similarity_scores:
            plot_similarity_distribution(similarity_scores, output_file=PLOT_FILE)
        else:
            print("유사도 값이 없어 시각화를 생략합니다.")

        result_df = pd.DataFrame(result_rows)
        result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"총 {len(queries)}개의 query 처리 완료.")
        print(f"저장 경로: {OUTPUT_CSV}")
        # print(result_df.head(10))

if __name__ == "__main__":
    semantic_search(FILE_TASKS)