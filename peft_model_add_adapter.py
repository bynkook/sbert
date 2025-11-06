"""
SBERT에 PEFT(LoRA) 어댑터를 부착하여 도메인 용어 사전으로 효율적 파인튜닝

개요
- 로컬 기반 모델('./ko-sroberta')을 SentenceTransformers의 Transformer 모듈로 로드한 뒤,
  LoRA 어댑터를 오토모델 레벨에 주입(get_peft_model)하여 파라미터 효율적 미세조정을 수행한다.
- Attention Q/K/V 모듈명의 변종(query/key/value vs q_proj/k_proj/v_proj)을 자동 탐지해 주입 실패를 방지한다.
- 학습 손실로 MultipleNegativesRankingLoss를 사용하여 용어–정의 쌍에서 강한 네거티브 샘플링 효과를 확보한다.
- 일반 성능 보전을 위해 KorSTS dev 세트로 EmbeddingSimilarityEvaluator를 함께 사용해 학습 중·후 평가를 수행한다.
- 학습 종료 후에는 전체 ST 모델이 아니라, LoRA 어댑터 가중치만 별도 경로(output_path)로 저장한다.

입력 데이터
- 'prepared_datasets' 폴더 내 전처리 산출물:
  - korsts_dev.pkl: 검증용 STS 샘플
  - dict_domain.pkl: 용어–정의 InputExample 리스트(라벨 불필요)

출력
- output_path 폴더에 PEFT(LoRA) 어댑터(예: adapter_config.json, adapter_model.bin)만 저장.

하이퍼파라미터
- epochs, batch_size, learning_rate, warmup_ratio
- lora_r, lora_alpha, lora_dropout

주의점
- 주입된 LoRA 파라미터 개수 점검으로 target_modules 미스매치를 조기에 탐지한다.
- SentenceTransformer 파이프라인 자체는 저장하지 않으므로, 추론 시에는 동일 베이스 모델을 로드하고
  저장된 어댑터를 붙인 뒤 동일 Pooling 설정으로 재구성해야 한다.
"""

import math
import os
import warnings
warnings.filterwarnings('ignore')
import pickle
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from peft import get_peft_model, LoraConfig, TaskType

def finetune_model_with_peft(
        dataset_path,
        output_path,
        epochs=4,
        batch_size=12,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05
        ):
    """
    PEFT(LoRA) 방식으로 SBERT를 도메인 용어 사전에 맞게 미세조정하고 어댑터만 저장한다.

    Args:
        dataset_path (str): 전처리된 KorSTS(dev)와 dict_domain.pkl이 위치한 폴더 경로.
        output_path (str): 학습된 LoRA 어댑터를 저장할 경로(폴더).
        epochs (int): 학습 에폭 수.
        batch_size (int): 배치 크기.
        learning_rate (float): 어댑터 학습 학습률.
        warmup_ratio (float): 웜업 스텝 비율.
        lora_r (int): LoRA 랭크.
        lora_alpha (int): LoRA 스케일.
        lora_dropout (float): LoRA 드롭아웃.

    Process:
        1) 로컬 베이스 모델('./ko-sroberta')을 Transformer로 로드.
        2) QKV 모듈명 자동 탐지 후 LoRA 어댑터 주입(get_peft_model).
        3) Pooling을 결합해 SentenceTransformer 파이프라인 구성.
        4) dict_domain.pkl로 MultipleNegativesRankingLoss 학습.
        5) KorSTS dev로 EmbeddingSimilarityEvaluator 평가.
        6) 어댑터 가중치만 output_path로 저장.

    Returns:
        None. (부수효과로 output_path에 어댑터 파일 저장, 학습 로그 출력)

    Notes:
        - SentenceTransformer 전체 모델은 저장하지 않는다.
        - 추론 시에는 동일 베이스 모델에 저장된 어댑터를 PeftModel.from_pretrained로 부착하고,
          학습과 동일한 Pooling 설정을 사용해야 일관된 임베딩/유사도가 보장된다.
    """
    # 1. 모델 로드
    print(f"[1] Base 모델 및 PEFT(LoRA) 설정")
    # SentenceTransformer 내부의 Transformer 모델에 직접 접근하여 어댑터 추가
    base_model_path = './ko-sroberta'    # use offline downloaded model
    word_embedding_model = models.Transformer(base_model_path)

    # --- QKV 모듈명 자동 탐지: (query,key,value) vs (q_proj,k_proj,v_proj) ---
    def infer_qkv_names(hf_model):
        names = set(n.split(".")[-1] for n, m in hf_model.named_modules() if hasattr(m, "weight"))
        if {"query","key","value"}.issubset(names):
            return ["query","key","value"]
        if {"q_proj","k_proj","v_proj"}.issubset(names):
            return ["q_proj","k_proj","v_proj"]
        # 드물게 attn.W_q 등 커스텀 네이밍을 대비
        candidates = [t for t in ["query","key","value","q_proj","k_proj","v_proj"] if t in names]
        if candidates:
            return candidates
        raise RuntimeError(f"ERROR: QKV 모듈명을 찾지 못했습니다. 실제 말단 모듈들: {sorted(list(names))[:20]} ...")
    
    # PEFT (LoRA) 설정
    target_qkv = infer_qkv_names(word_embedding_model.auto_model)
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION, # SBERT는 특징 추출기로 사용
        target_modules=target_qkv              # Attention Q,K,V에 LoRA 적용
    )
    
    # PEFT 모델로 변환
    word_embedding_model.auto_model = get_peft_model(word_embedding_model.auto_model, peft_config)
    # 주입 검증: LoRA 파라미터가 없으면 target_modules 미스매치
    word_embedding_model.auto_model.print_trainable_parameters()
    injected = [n for n, p in word_embedding_model.auto_model.named_parameters() if "lora_" in n and p.requires_grad]
    if len(injected) == 0:
        raise RuntimeError("ERROR: LoRA 모듈이 주입되지 않았습니다. target_modules 설정을 확인하세요.")

    # Pooling 레이어 추가하여 SBERT 모델 완성
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # 2. 데이터셋 준비
    print("[2] 학습 데이터셋 준비")
    # 2-1. 저장된 KorSTS 데이터셋 로드 (평가용)
    print(f"  - '{dataset_path}'에서 전처리된 데이터셋 로드...")
    sts_dev_path = os.path.join(dataset_path, "korsts_dev.pkl")
    with open(sts_dev_path, "rb") as f:
        sts_dev = pickle.load(f)
    print(f"    KorSTS 검증 샘플: {len(sts_dev)}개 (평가용으로만 사용)")

    # 2-2. 저장된 전문 용어 사전 데이터셋 로드 (학습용)
    dict_path = os.path.join(dataset_path, "dict_domain.pkl")
    with open(dict_path, "rb") as f:
        dict_domain = pickle.load(f)
    print(f"    전문 용어 사전 샘플: {len(dict_domain)}개")

    # 3. DataLoader 및 Loss 정의
    print("[3] DataLoader 및 Loss 정의")
    # 전문 용어 사전용 DataLoader 및 Loss
    dict_dataloader = DataLoader(dict_domain, shuffle=True, batch_size=batch_size)
    # 어댑터 학습에는 MultipleNegativesRankingLoss가 매우 효과적입니다.
    dict_loss = losses.MultipleNegativesRankingLoss(model)
    print("  - 전문 용어 사전용 Loss: MultipleNegativesRankingLoss")

    # 4. 모델 평가자(Evaluator) 준비
    print("[4] 모델 평가자 준비")
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(sts_dev, name='sts-dev')
    print("  - KorSTS 검증셋으로 EmbeddingSimilarityEvaluator 생성")

    # 5. 모델 fine-tuning 실행
    print("[5] PEFT(LoRA) 파인튜닝 시작")
    steps_per_epoch = len(dict_dataloader)
    warmup_steps = math.ceil(steps_per_epoch * epochs * warmup_ratio)
    print(f"  - 학습 파라미터: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, warmup_steps={warmup_steps}")

    model.fit(
        train_objectives=[(dict_dataloader, dict_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        optimizer_params={'lr': learning_rate},
    )
    
    # 학습된 PEFT(LoRA) 어댑터만 저장
    word_embedding_model.auto_model.save_pretrained(output_path)

    print(f"\n--- 파인튜닝 완료 ---")
    print(f"학습된 PEFT 어댑터 저장 경로: {output_path}")
    
    # 6. 최종 모델 성능 평가
    print("[6] 최종 모델 성능 평가")
    final_score = evaluator(model)
    print(final_score)


if __name__ == '__main__':
    # 1. 전처리된 데이터셋이 저장된 폴더 경로
    preprocessed_dataset_path = 'prepared_datasets'

    # 2. 학습된 어댑터를 저장할 경로 지정
    peft_output_path = 'output/ko-sroberta-lora'

    # 3. Fine-tuning 함수 호출
    finetune_model_with_peft(
        dataset_path=preprocessed_dataset_path,
        output_path=peft_output_path,
        epochs=8,
        batch_size=16,
        learning_rate=1e-4
    )
