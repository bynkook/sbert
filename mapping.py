# %%
'''import modules'''
import re
import pandas as pd
from icecream import ic
from pathlib import Path
OUTPUT_DIR = Path("./data")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %%
'''load tables'''
df_func1 = pd.read_csv(OUTPUT_DIR / 'function_건축.csv', sep='\t')
df_func2 = pd.read_csv(OUTPUT_DIR / 'function_토목.csv', sep='\t')
df_func3 = pd.read_csv(OUTPUT_DIR / 'function_플랜트.csv', sep='\t')
df_func4 = pd.read_csv(OUTPUT_DIR / 'function_주택.csv', sep='\t')
df_func5 = pd.read_csv(OUTPUT_DIR / 'function_하이테크.csv', sep='\t')

df_proc1 = pd.read_csv(OUTPUT_DIR / 'process_건축.csv', sep='\t')
df_proc2 = pd.read_csv(OUTPUT_DIR / 'process_토목.csv', sep='\t')
df_proc3 = pd.read_csv(OUTPUT_DIR / 'process_플랜트.csv', sep='\t')
df_proc4 = pd.read_csv(OUTPUT_DIR / 'process_주택.csv', sep='\t')
df_proc5 = pd.read_csv(OUTPUT_DIR / 'process_하이테크.csv', sep='\t')

# %%
'''print head(3)'''
ic(df_func5.head(3))
ic(df_proc5.head(3))

# %%
'''컬럼명 소문자로 변경'''
def column_names_to_lowercase(df):
    df.columns = df.columns.astype(str).str.lower()
    return df

df_func1 = column_names_to_lowercase(df_func1)
df_func2 = column_names_to_lowercase(df_func2)
df_func3 = column_names_to_lowercase(df_func3)
df_func4 = column_names_to_lowercase(df_func4)
df_func5 = column_names_to_lowercase(df_func5)

df_proc1 = column_names_to_lowercase(df_proc1)
df_proc2 = column_names_to_lowercase(df_proc2)
df_proc3 = column_names_to_lowercase(df_proc3)
df_proc4 = column_names_to_lowercase(df_proc4)
df_proc5 = column_names_to_lowercase(df_proc5)

# %%
'''문자 대치, 삭제'''
def replace_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """개행 문자, 공백, 콤마 앞 공백 정리"""
    df_copy = df.copy()
    for col in df_copy.columns:
        if pd.api.types.is_string_dtype(df_copy[col].dtype):
            df_copy[col] = (df_copy[col].replace(r"\r\n|\\n|\\N", " ", regex=True)    # 개행문자 공백으로
                                        .replace(r"\s+", " ", regex=True)             # 여러 공백을 하나로
                                        .replace(r"\s*,\s*", ", ", regex=True)        # 콤마 앞,뒤 공백 정리                                        
                                        .replace(r"(.+?,\s*)\1+", r"\1", regex=True)  # 연속 중복 단어 제거 (예: "A, A, " -> "A, ")
                                        .replace(r",(\s*),", ", ", regex=True)        # 공백으로 연결된 콤마 정리                                        
                                        .str.strip()                                  # 양쪽 공백 제거
                                        .str.rstrip(','))                             # 마지막 콤마 제거
    return df_copy

def clean_null_synonyms(df: pd.DataFrame) -> pd.DataFrame:
    """'NA', 'NULL' 등의 문자열을 공백으로 변경(csv 출력용)"""
    df_copy = df.copy()
    remove_patterns = re.compile(r"\b(N/?A|NaN|NONE|NULL)\b", flags=re.IGNORECASE)
    cols_to_clean = df_copy.columns.difference(['menu_id', 'program_id'])   # menu_id, program_id 는 결측치 표현 유지해야함.
    
    for col in cols_to_clean:
        if pd.api.types.is_string_dtype(df_copy[col].dtype):
            df_copy[col] = (df_copy[col].astype(str).replace(remove_patterns, " ", regex=True)
                                        .str.strip())
    return df_copy

def handle_special_columns(df: pd.DataFrame) -> pd.DataFrame:
    """'menu_id', 'program_id' 컬럼의 빈 문자열을 None으로 변환(.dropna 처리용)"""
    df_copy = df.copy()
    for col in ['menu_id', 'program_id']:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].replace(r"^\s*$", None, regex=True)
    return df_copy

def text_cleaner(df: pd.DataFrame) -> pd.DataFrame:
    """데이터 정제 파이프라인을 실행"""
    return df.pipe(replace_whitespace).pipe(clean_null_synonyms).pipe(handle_special_columns).pipe(replace_whitespace)

df_func1 = text_cleaner(df_func1)
df_func2 = text_cleaner(df_func2)
df_func3 = text_cleaner(df_func3)
df_func4 = text_cleaner(df_func4)
df_func5 = text_cleaner(df_func5)

df_proc1 = text_cleaner(df_proc1)
df_proc2 = text_cleaner(df_proc2)
df_proc3 = text_cleaner(df_proc3)
df_proc4 = text_cleaner(df_proc4)
df_proc5 = text_cleaner(df_proc5)

# %%
'''결측치 확인'''
print(df_func5.isna().sum())
print(df_proc5.isna().sum())

# %%
'''columns for sentence_transformer input'''
def build_query_table(df):

    # 선택사항 : menu_id, program_id 둘 중 1개만 없어도 그 행은 삭제
    df = df.dropna(subset=['menu_id', 'program_id'], axis=0, how='any')

    return df[['system',
               'menu_lv1_nm',
               'menu_lv2_nm',
               'menu_lv3_nm',
               'menu_lv4_nm',
               'menu_lv5_nm',
               '업무기능',
               'context',
               'menu_id',
               'program_id'
               ]]

df_query1 = build_query_table(df_func1)
df_query2 = build_query_table(df_func2)
df_query3 = build_query_table(df_func3)
df_query4 = build_query_table(df_func4)
df_query5 = build_query_table(df_func5)

# %%
'''결측치 확인'''
print(df_func5.info())
print(df_query5.info())
print(df_query5.isnull().sum())

# %%
'''테이블 shape[0] 저장'''
nrows_before = [
    (df_query1.shape[0], df_proc1.shape[0]),
    (df_query2.shape[0], df_proc2.shape[0]),
    (df_query3.shape[0], df_proc3.shape[0]),
    (df_query4.shape[0], df_proc4.shape[0]),
    (df_query5.shape[0], df_proc5.shape[0]),
    ]

# %%
'''중복행 첫번째 행으로 통합하고 이후 나머지 삭제'''
df_query1.drop_duplicates(inplace=True)
df_query2.drop_duplicates(inplace=True)
df_query3.drop_duplicates(inplace=True)
df_query4.drop_duplicates(inplace=True)
df_query5.drop_duplicates(inplace=True)

df_proc1.drop_duplicates(inplace=True)
df_proc2.drop_duplicates(inplace=True)
df_proc3.drop_duplicates(inplace=True)
df_proc4.drop_duplicates(inplace=True)
df_proc5.drop_duplicates(inplace=True)

nrows_after = [
    (df_query1.shape[0], df_proc1.shape[0]),
    (df_query2.shape[0], df_proc2.shape[0]),
    (df_query3.shape[0], df_proc3.shape[0]),
    (df_query4.shape[0], df_proc4.shape[0]),
    (df_query5.shape[0], df_proc5.shape[0]),
    ]

# 중복행 제거되는지 확인
for i in range(5):
    print(f'Function Table_{i+1} 중복제거후 행 갯수 차이(당초-변경) = {nrows_before[i][0]-nrows_after[i][0]}')
    print(f'Process  Table_{i+1} 중복제거후 행 갯수 차이(당초-변경) = {nrows_before[i][1]-nrows_after[i][1]}')

# %%
'''csv 파일 새로 저장'''
df_query1.to_csv(OUTPUT_DIR / 'function_건축_.csv', encoding='utf-8-sig', index=False)
df_query2.to_csv(OUTPUT_DIR / 'function_토목_.csv', encoding='utf-8-sig', index=False)
df_query3.to_csv(OUTPUT_DIR / 'function_플랜트_.csv', encoding='utf-8-sig', index=False)
df_query4.to_csv(OUTPUT_DIR / 'function_주택_.csv', encoding='utf-8-sig', index=False)
df_query5.to_csv(OUTPUT_DIR / 'function_하이테크_.csv', encoding='utf-8-sig', index=False)

df_proc1.to_csv(OUTPUT_DIR / 'process_건축_.csv', encoding='utf-8-sig', index=False)
df_proc2.to_csv(OUTPUT_DIR / 'process_토목_.csv', encoding='utf-8-sig', index=False)
df_proc3.to_csv(OUTPUT_DIR / 'process_플랜트_.csv', encoding='utf-8-sig', index=False)
df_proc4.to_csv(OUTPUT_DIR / 'process_주택_.csv', encoding='utf-8-sig', index=False)
df_proc5.to_csv(OUTPUT_DIR / 'process_하이테크_.csv', encoding='utf-8-sig', index=False)
# %%
