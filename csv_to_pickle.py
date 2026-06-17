import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sys
import os

# =========================================================
# ⚙️ [설정] 파일 경로 지정
# =========================================================
OUTPUT_DIR = 'output'  # 저장할 폴더 이름
DATA_DIR = 'data'      # csv 원본 데이터 파일 저장 폴더

# output 폴더가 없으면 에러 (안전장치)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"📂 '{OUTPUT_DIR}' 폴더를 생성했습니다.")

CSV_PATH = os.path.join(DATA_DIR, 'movie_data.csv')

# =========================================================
# 1. 데이터 로드 및 전처리
# =========================================================
print("⏳ [System] 데이터 로드 중...")
try:
    data = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"❌ [Error] '{CSV_PATH}' 파일을 찾을 수 없습니다.")
    sys.exit(1)

data = data.fillna('')

def create_soup(x):
    # 장르, 감독, 영화명 등을 합쳐서 하나의 풍부한 텍스트로 만듦
    # return str(x['장르']) + ' ' + str(x['감독']) + ' ' + str(x['영화명'])
    return str(x['장르'])

data['soup'] = data.apply(create_soup, axis=1)

# =========================================================
# 2. 벡터화 및 유사도 계산
# =========================================================
print("🔢 [System] 벡터화 및 코사인 유사도 계산 중...")

# (1) Vectorizer 생성 및 학습
count = CountVectorizer(stop_words=None)
count_matrix = count.fit_transform(data['soup'])

# =========================================================
# 3. 데이터 저장 (Pickle) -> output 폴더에 저장
# =========================================================
print(f"💾 [System] '{OUTPUT_DIR}' 폴더에 피클 파일 저장 중...")

try:
    # 1. 영화 정보 (Metadata)
    with open(os.path.join(OUTPUT_DIR, 'metadata.pickle'), 'wb') as f:
        pickle.dump(data, f)
    
    # 2. 벡터화 도구 (사용자 감정 -> 숫자 변환용, 필수!)
    with open(os.path.join(OUTPUT_DIR, 'vectorizer.pickle'), 'wb') as f:
        pickle.dump(count, f)

    # 3. 영화 특성 매트릭스 (감정 벡터와 비교용, 필수!)
    with open(os.path.join(OUTPUT_DIR, 'count_matrix.pickle'), 'wb') as f:
        pickle.dump(count_matrix, f)


    print("✅ [Success] 모든 파일 저장 완료!")
    print(f"   - {os.path.join(OUTPUT_DIR, 'metadata.pickle')}")
    print(f"   - {os.path.join(OUTPUT_DIR, 'vectorizer.pickle')}")
    print(f"   - {os.path.join(OUTPUT_DIR, 'count_matrix.pickle')}")

except Exception as e:
    print(f"❌ [Error] 파일 저장 실패: {e}")