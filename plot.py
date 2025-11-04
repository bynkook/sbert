import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_similarity_distribution(scores, output_file):
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
    plt.figure(figsize=(10, 6))
    bar = plt.bar(labels, hist, color='skyblue', edgecolor='black')
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.1f' % height, ha='center', va='bottom', size = 9)

    plt.title('유사도 값 분포', fontsize=12, pad=10)
    # plt.xlabel('유사도 구간', fontsize=10)
    # plt.ylabel('빈도', fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # 그래프 저장
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"유사도 분포 그래프 저장 완료: {output_file}")

if __name__ == "__main__":
    # 엑셀 원데이터 점수와 비교용 plot
    score_excel = pd.read_csv('score_excel.csv')
    plot_file = "./similarity_distribution_excel.png"
    plot_similarity_distribution(score_excel, output_file=plot_file)