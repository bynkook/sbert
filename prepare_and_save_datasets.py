"""
KorSTS와 도메인 용어 사전을 SentenceTransformers 학습용 포맷으로 전처리하여 저장

개요
- Korpora에서 KorSTS를 로드하고, 유사도 점수를 0~1 범위로 정규화한 InputExample로 변환한다.
  (train 세트는 학습용, dev 세트는 평가용으로 사용)
- 'data/domain_dictionary.csv'에서 ['term', 'definition']를 읽어 용어–정의 쌍 InputExample을 생성한다.
  (MultipleNegativesRankingLoss 학습에 사용, 라벨 불필요)
- 세 개의 pickle 파일을 output_folder에 저장:
  - korsts_train.pkl, korsts_dev.pkl, dict_domain.pkl

출력 파일
- {output_folder}/korsts_train.pkl
- {output_folder}/korsts_dev.pkl
- {output_folder}/dict_domain.pkl

의존성
- Korpora, pandas, sentence-transformers

주의점
- Korpora.load('korsts', root_dir='./Korpora') 사용. 로컬에 파일이 있으면 재다운로드를 피하도록 유틸을 수정한 환경을 가정한다.
- domain_dictionary.csv는 최소 ['term','definition'] 컬럼이 필요하다.
"""

import pickle
import pandas as pd
from Korpora import Korpora
from sentence_transformers import InputExample

def prepare_and_save_datasets(output_folder="."):
    """
    KorSTS와 도메인 용어 사전을 로드하여 SentenceTransformers 학습/평가용 InputExample 리스트로 변환하고 pickle로 저장한다.

    Args:
        output_folder (str): 산출물을 저장할 대상 폴더 경로. 존재하지 않으면 사전에 생성해야 한다.

    Process:
        1) Korpora.load('korsts')로 KorSTS 로드.
           - train → 점수/5.0 정규화 후 InputExample(texts=[s1, s2], label=score)
           - dev   → 동일 방식으로 검증 세트 생성
        2) CSV 'data/domain_dictionary.csv' 로드 후 dict(term→definition) 구성.
           - InputExample(texts=[term, definition]) 목록 생성(라벨 불필요)
        3) 세 개의 pickle 파일로 저장:
           - korsts_train.pkl, korsts_dev.pkl, dict_domain.pkl

    Returns:
        None. (output_folder에 세 개의 .pkl 파일 저장)

    Raises:
        FileNotFoundError: domain_dictionary.csv를 찾지 못한 경우.

    Notes:
        - KorSTS train은 현재 스크립트에서 직접 학습에 사용하지 않더라도, 추후 확장 시 재사용 가능하도록 함께 저장한다.
        - 도메인 사전은 MNRL 학습에서 배치 내 부정 샘플로 작동하므로 규모가 클수록 효과적이다.
    """
    print("--- 데이터셋 준비 및 저장 시작 ---")

    # 1. KorSTS 데이터셋 준비 (학습용, 검증용)
    # --------------------------------------------------
    print("\n[1/3] KorSTS 데이터셋 로드 및 전처리 중...")
    try:
        # Korpora를 통해 데이터셋을 fetch하고 load합니다.
        # Korpora.fetch('korsts', root_dir='./Korpora')
        corpus = Korpora.load('korsts', root_dir='./Korpora')
        '''지정된 경로에 파일이 있으면 다운로드 하지 않도록 utils.py 를 수정했음'''

        # 학습 데이터 (Train set)
        sts_train = []
        for example in corpus.train:
            score = float(example.label) / 5.0  # 점수를 0.0 ~ 1.0 사이로 정규화
            sts_train.append(InputExample(texts=[example.text, example.pair], label=score))        
        print(f"  - KorSTS 학습 샘플 {len(sts_train)}개 생성 완료.")

        # 검증 데이터 (Validation set) - 모델 성능 평가용
        sts_dev = []
        for example in corpus.dev:
            score = float(example.label) / 5.0
            sts_dev.append(InputExample(texts=[example.text, example.pair], label=score))        
        print(f"  - KorSTS 검증 샘플 {len(sts_dev)}개 생성 완료.")

    except Exception as e:
        print(f"오류: KorSTS 데이터셋 로드에 실패했습니다. 인터넷 연결 또는 Korpora 라이브러리를 확인하세요. ({e})")
        pass

    # 2. 전문 용어 사전 데이터셋 준비
    # --------------------------------------------------
    print("\n[2/3] 전문 용어 사전 데이터셋 전처리 중...")
    
    csv_path = "data/domain_dictionary.csv"    # csv 파일은 'term', 'definition' 두 열로 구성
    try:
        df_dict = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    # DataFrame → dict 변환
    domain_dict = dict(zip(df_dict["term"], df_dict["definition"]))

    print(f"전문 용어 {len(domain_dict)}개 로드 완료.")

    # MultipleNegativesRankingLoss는 라벨이 필요 없으며, (anchor, positive) 쌍만 있으면 됩니다.
    dict_domain = []
    for term, definition in domain_dict.items():
        dict_domain.append(InputExample(texts=[term, definition]))
    
    print(f"  - 전문 용어 사전 샘플 {len(dict_domain)}개 생성 완료.")

    # 3. 전처리된 데이터셋 파일로 저장
    # --------------------------------------------------
    print("\n[3/3] 전처리된 데이터셋을 파일로 저장 중...")
    
    # 각 데이터셋을 별도의 pickle 파일로 저장
    sts_train_path = f"{output_folder}/korsts_train.pkl"
    with open(sts_train_path, "wb") as f:
        pickle.dump(sts_train, f)
    print(f"  - KorSTS 학습 데이터 저장 완료: {sts_train_path}")

    sts_dev_path = f"{output_folder}/korsts_dev.pkl"
    with open(sts_dev_path, "wb") as f:
        pickle.dump(sts_dev, f)
    print(f"  - KorSTS 검증 데이터 저장 완료: {sts_dev_path}")

    dict_path = f"{output_folder}/dict_domain.pkl"
    with open(dict_path, "wb") as f:
        pickle.dump(dict_domain, f)
    print(f"  - 전문 용어 사전 데이터 저장 완료: {dict_path}")

    print("\n--- 모든 데이터셋 준비 및 저장이 완료되었습니다. ---")


if __name__ == '__main__':
    # 저장할 폴더를 지정합니다. 예: 'datasets'
    # 폴더가 존재하지 않으면 미리 생성해야 합니다.
    import os
    output_directory = "prepared_datasets"
    os.makedirs(output_directory, exist_ok=True)
    
    prepare_and_save_datasets(output_folder=output_directory)
