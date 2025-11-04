import math
import os
import warnings
warnings.filterwarnings('ignore')
import pickle
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from peft import get_peft_model, LoraConfig, TaskType

def finetune_expert_model_with_peft(
        dataset_path,
        output_path,
        epochs=1,
        batch_size=16,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05
        ):
    """
    PEFT(LoRA)를 사용하여 SBERT 모델을 전문 용어 사전으로 효율적으로 파인튜닝합니다.
    
    **재훈련 방법론: PEFT (Parameter-Efficient Fine-Tuning) with LoRA**
    
    이 코드는 기존의 Full fine-tuning 방식 대신, '어댑터' 기반의 파라미터 효율적 파인튜닝(PEFT) 방식을 사용합니다.
    
    **목적 및 채택 이유:**
    기반 모델('jhgan/ko-sroberta-multitask')은 이미 KorSTS와 같은 일반적인 한국어 데이터로 충분히 학습되어 있습니다.
    우리의 목표는 이 모델이 가진 기존의 범용 언어 이해 능력을 훼손하지 않으면서, '전문 용어'라는 새로운 도메인 지식만
    빠르고 효율적으로 추가하는 것입니다. 어댑터 방식은 이러한 목적에 가장 부합합니다.
    
    **장점:**
    1.  **압도적으로 빠른 학습 속도**: 사전 학습된 모델의 수십억 개 파라미터는 모두 동결(freeze)하고, 모델 중간에 삽입된
        아주 작은 규모의 어댑터 모듈(수백만 개 파라미터)만 학습합니다. 이로 인해 재훈련 시간이 크게 단축됩니다.
    2.  **치명적 망각(Catastrophic Forgetting) 원천 방지**: 기존 모델의 가중치를 전혀 수정하지 않으므로, 원래 모델이
        가지고 있던 일반적인 문장 유사도 측정 능력이 저하될 위험이 없습니다.
    3.  **효율적인 모델 관리**: 전체 모델(수백 MB)을 복제하여 저장할 필요 없이, 학습된 어댑터(수 MB)만 별도로 저장하면
        됩니다. 이를 통해 다양한 도메인별 어댑터를 손쉽게 관리하고 교체하며 사용할 수 있습니다.
    
    **의존성 (필요 패키지):**
    - pip install sentence-transformers transformer[torch] peft
    
    Args:
        model_name (str): fine-tuning할 기반 모델 이름
        dataset_path (str): 전처리된 데이터셋이 저장된 폴더 경로
        output_path (str): 학습된 모델을 저장할 경로
        epochs (int): 학습 에폭 수
        batch_size (int): 배치 사이즈
        learning_rate (float): 어댑터 학습을 위한 학습률 (일반적으로 fine-tuning보다 높게 설정)
        warmup_ratio (float): 전체 학습 스텝 대비 웜업 스텝의 비율
        lora_r (int): LoRA의 rank.
        lora_alpha (int): LoRA scaling factor.
        lora_dropout (float): LoRA 레이어의 드롭아웃 비율.
    """
    # 1. 모델 로드
    print(f"--- 1. Base 모델 및 PEFT(LoRA) 설정 ---")
    # SentenceTransformer 내부의 Transformer 모델에 직접 접근하여 어댑터 추가
    base_model_path = './ko-sroberta'    # use offline downloaded model
    word_embedding_model = models.Transformer(base_model_path)
    
    # PEFT (LoRA) 설정
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION, # SBERT는 특징 추출기로 사용되므로
        target_modules=["query", "key", "value"] # Attention 레이어의 Q, K, V에 LoRA 적용
    )
    
    # PEFT 모델로 변환
    word_embedding_model.auto_model = get_peft_model(word_embedding_model.auto_model, peft_config)
    word_embedding_model.auto_model.print_trainable_parameters()

    # Pooling 레이어 추가하여 SBERT 모델 완성
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    print("-" * 50)

    # 2. 데이터셋 준비
    print("--- 2. 학습 데이터셋 준비 ---")
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
    print("-" * 50)

    # 3. DataLoader 및 Loss 정의
    print("--- 3. DataLoader 및 Loss 정의 ---")
    # 전문 용어 사전용 DataLoader 및 Loss
    dict_dataloader = DataLoader(dict_domain, shuffle=True, batch_size=batch_size)
    # 어댑터 학습에는 MultipleNegativesRankingLoss가 매우 효과적입니다.
    dict_loss = losses.MultipleNegativesRankingLoss(model)
    print("  - 전문 용어 사전용 Loss: MultipleNegativesRankingLoss")
    print("-" * 50)

    # 4. 모델 평가자(Evaluator) 준비
    print("--- 4. 모델 평가자 준비 ---")
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(sts_dev, name='sts-dev')
    print("  - KorSTS 검증셋으로 EmbeddingSimilarityEvaluator 생성")
    print("-" * 50)

    # 5. 모델 fine-tuning 실행
    print("--- 5. PEFT(LoRA) 파인튜닝 시작 ---")
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
        save_best_model=True,
        output_path=output_path
    )
    
    # 학습된 PEFT(LoRA) 어댑터만 저장
    word_embedding_model.auto_model.save_pretrained(output_path)

    print(f"\n--- 파인튜닝 완료 ---")
    print(f"학습된 PEFT 어댑터 저장 경로: {output_path}")
    
    # 6. 최종 모델 성능 평가
    print("\n--- 6. 최종 모델 성능 평가 ---")
    final_score = evaluator(model)
    print(final_score)


if __name__ == '__main__':
    # 1. 전처리된 데이터셋이 저장된 폴더 경로
    preprocessed_dataset_path = 'prepared_datasets'

    # 2. 학습된 어댑터를 저장할 경로 지정
    peft_output_path = 'output/ko-sroberta-expert-lora'

    # 3. Fine-tuning 함수 호출
    finetune_expert_model_with_peft(
        dataset_path=preprocessed_dataset_path,
        output_path=peft_output_path,
        epochs=50,
        batch_size=16,
        learning_rate=1e-4
    )
