from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types
from dotenv import load_dotenv
import requests
import json
import random
import pickle
import os

load_dotenv()  # .env 파일에서 환경 변수 로드
import subprocess
import sys

app = Flask(__name__)

TMDB_API_KEY = os.getenv('TMDB_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

#Gemini 클라이언트 설정
client = None
if GOOGLE_API_KEY and '여기에' not in GOOGLE_API_KEY:
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        print("✅ Gemini 연결 성공")
    except Exception as e:
        print(f"⚠️ 클라이언트 설정 오류: {e}")

# =========================================================
#  [데이터 로드]
# =========================================================
df = None
vectorizer = None
count_matrix = None
ALL_GENRES = []

def extract_unique_genres(dataframe):
    genres = set()
    try:
        for raw_genre in dataframe['장르']:
            if pd.isna(raw_genre): continue
            parts = str(raw_genre).replace(',', ' ').split()
            for p in parts:
                if p.strip():
                    genres.add(p.strip())
        return list(genres)
    except Exception as e:
        print(f"⚠️ [Warning] 장르 추출 실패: {e}")
        return []

def check_and_generate_pickle_files():
    """
    output 폴더에 필요한 pickle 파일 3개가 모두 존재하는지 확인.
    하나라도 없으면 csv_to_pickle.py를 실행하여 생성.
    """
    base_path = 'output'
    required_files = ['metadata.pickle', 'vectorizer.pickle', 'count_matrix.pickle']

    # 필요한 파일들이 모두 존재하는지 확인
    missing_files = []
    for file_name in required_files:
        file_path = os.path.join(base_path, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)

    # 파일이 하나라도 없으면 csv_to_pickle.py 실행
    if missing_files:
        print(f"⚠️ [System] 필요한 pickle 파일이 존재하지 않습니다: {', '.join(missing_files)}")
        print(f"🔄 [System] csv_to_pickle.py를 실행하여 pickle 파일을 생성합니다...")

        try:
            # csv_to_pickle.py 실행 (UTF-8 인코딩 설정)
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            result = subprocess.run(
                [sys.executable, 'csv_to_pickle.py'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                check=True,
                env=env
            )

            # 실행 결과 출력
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)

            print(f"✅ [System] pickle 파일 생성 완료!")

        except subprocess.CalledProcessError as e:
            print(f"❌ [Error] csv_to_pickle.py 실행 실패: {e}")
            print(f"   stdout: {e.stdout}")
            print(f"   stderr: {e.stderr}")
            sys.exit(1)
        except FileNotFoundError:
            print(f"❌ [Error] csv_to_pickle.py 파일을 찾을 수 없습니다.")
            sys.exit(1)
    else:
        print(f"✅ [System] 필요한 pickle 파일이 모두 존재합니다.")

def load_data():
    global df, vectorizer, count_matrix, ALL_GENRES
    base_path = 'output'
    print(f"📂 [System] '{base_path}' 폴더에서 데이터 로드 중...")

    try:
        with open(os.path.join(base_path, 'metadata.pickle'), 'rb') as f:
            df = pickle.load(f)
        with open(os.path.join(base_path, 'vectorizer.pickle'), 'rb') as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(base_path, 'count_matrix.pickle'), 'rb') as f:
            count_matrix = pickle.load(f)

        ALL_GENRES = extract_unique_genres(df)
        print(f"✅ [System] 데이터 로드 완료! (장르 {len(ALL_GENRES)}개 로드됨)")

    except FileNotFoundError as e:
        print(f"❌ [Error] 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"❌ [Error] 데이터 로드 중 오류 발생: {e}")

# pickle 파일 검증 및 생성
check_and_generate_pickle_files()

# 데이터 로드
load_data()

# =========================================================
# 🛠️ [기능 함수들]
# =========================================================

def get_tmdb_info(title):
    """
    TMDB API를 사용하여 포스터, 줄거리, 그리고 감독 정보를 가져오는 함수
    """
    fallback_poster = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/No_image_available.svg/300px-No_image_available.svg.png"
    
    if not TMDB_API_KEY or '여기에' in TMDB_API_KEY: 
        return fallback_poster, "정보 없음", "감독 미상"
    
    try:
        # 1. 영화 검색
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}&language=ko-KR"
        res = requests.get(url, timeout=2).json()
        
        if res.get('results'):
            m = res['results'][0]
            movie_id = m['id']
            poster = f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get('poster_path') else fallback_poster
            plot = m.get('overview', "")

            # 2. 감독 정보 가져오기 (Credits API)
            credit_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={TMDB_API_KEY}&language=ko-KR"
            credit_res = requests.get(credit_url, timeout=2).json()
            
            director = "정보 없음"
            if 'crew' in credit_res:
                for person in credit_res['crew']:
                    if person['job'] == 'Director':
                        director = person['name']
                        break
            
            return poster, plot, director
            
    except Exception as e:
        print(f"⚠️ [TMDB Error] {e}")
        pass
        
    return fallback_poster, "", "정보 없음"

def get_keywords_from_gemini(user_emotions):
    if not client:
        return " ".join(user_emotions)

    emotion_str = ", ".join(user_emotions)
    available_genres_str = ", ".join(ALL_GENRES)

    prompt = f"""
    [역할]
    너는 영화 추천 시스템의 검색 엔진이야.

    [상황]
    사용자가 현재 느끼는 감정: "{emotion_str}"

    [임무]
    위 감정을 공감하기 위해 사용자가 봐야 할 영화의 '장르'나 '키워드'를 아래 제공된 [가능한 장르 목록] 중에서 골라줘.

    [가능한 장르 목록]
    {available_genres_str}

    [조건]
    1. 사용자의 감정에 가장 잘 어울리는 장르를 3개~5개 선택해.
    2. 답변은 다른 설명 없이 오직 '선택한 장르 단어들'만 띄어쓰기로 구분해서 적어줘.
    3. 예시: "액션 코미디 모험"
    """

    try:
        model_name = 'gemini-2.5-flash'
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        keywords = response.text.strip()
        print(f"🤖 [Gemini Keyword] {keywords}")
        return keywords

    except Exception as e:
        print(f"⚠️ [Gemini Error] 키워드 생성 실패: {e}")
        return emotion_str


def get_content_based_recommendations(search_keywords, start_year, end_year, page=1):
    if vectorizer is None or count_matrix is None:
        return []

    print(f"🔍 [Search] Page: {page}, 키워드: {search_keywords}, 기간: {start_year}~{end_year}")

    try:
        user_vector = vectorizer.transform([search_keywords])
    except ValueError:
        return []

    sim_scores = cosine_similarity(user_vector, count_matrix).flatten()
    sim_indices = sim_scores.argsort()[::-1]

    results = []
    skip_count = (page - 1) * 5
    skipped = 0

    for idx in sim_indices:
        if sim_scores[idx] < 0.01: continue

        row = df.iloc[idx]
        try:
            movie_year = int(row['제작연도'])
        except:
            movie_year = 0

        if movie_year < start_year or movie_year > end_year:
            continue

        if skipped < skip_count:
            skipped += 1
            continue

        results.append({
            "title": row['영화명'],
            "genre": row['장르'],
            "year": movie_year,
            "similarity": round(sim_scores[idx] * 100, 1)
        })

        if len(results) >= 5: break

    if not results and page == 1:
        print("⚠️ [System] 조건에 맞는 영화 없음. 랜덤 추천.")
        random_indices = random.sample(range(len(df)), min(5, len(df)))
        for idx in random_indices:
            row = df.iloc[idx]
            try: y = int(row['제작연도'])
            except: y = 0
            results.append({
                "title": row['영화명'],
                "genre": row['장르'],
                "year": y,
                "similarity": 0
            })

    return results

def ask_gemini_for_reason_and_food(emotions, movie_list):
    if not movie_list: return []
    if not client: return movie_list

    movies_str = ", ".join([m['title'] for m in movie_list])
    emotions_str = ", ".join(emotions)

    prompt = f"""
    상황: 사용자는 지금 '{emotions_str}'의 감정/기분을 느끼고 있습니다.
    [임무] 아래 영화들에 대해 JSON 형식으로 답변하세요.
    1. reason: 사용자가 느끼는 '{emotions_str}' 감정에 맞춰 이 영화를 추천하는 이유 (한국어 1문장)
    2. food: 영화 보면서 먹기 좋은 음식 1개

    [영화 목록]
    {movies_str}

    [응답 형식 - JSON List]
    [
      {{"title": "영화제목", "reason": "이유...", "food": "음식..."}}
    ]
    """

    try:
        model_name = 'gemini-2.5-flash'
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type='application/json')
        )
        ai_data = json.loads(response.text)

        for m in movie_list:
            ai_info = next((item for item in ai_data if item.get("title") == m['title']), None)
            if ai_info:
                m['reason'] = ai_info.get('reason', 'AI 추천작')
                m['food'] = ai_info.get('food', '팝콘')
            else:
                m['reason'] = 'AI 추천 영화입니다.'
                m['food'] = '팝콘'
    except Exception as e:
        print(f"⚠️ [Gemini Error] {e}. 기본값 사용.")
        for m in movie_list:
            m['reason'] = "추천 영화입니다."
            m['food'] = "팝콘"

    return movie_list

# =========================================================
# 🌐 [Flask 라우트]
# =========================================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend_by_emotion', methods=['POST'])
def recommend_by_emotion():
    import time
    
    # ⏱️ 전체 시간 측정 시작
    total_start = time.time()
    
    try:
        data = request.json
        emotions = data.get('emotions', [])
        start_year = int(data.get('start_year', 1900))
        end_year = int(data.get('end_year', 2025))
        page = int(data.get('page', 1))
        existing_keywords = data.get('keywords', None)

        if not emotions:
            return jsonify({"status": "error", "message": "감정을 선택해주세요."})

        # ⏱️ 1. Gemini API (키워드 생성) 시간 측정
        gemini_keyword_time = 0
        if existing_keywords:
            search_keywords = existing_keywords
            print(f"♻️ [System] 기존 키워드 재사용: {search_keywords}")
        else:
            gemini_keyword_start = time.time()
            search_keywords = get_keywords_from_gemini(emotions)
            gemini_keyword_time = time.time() - gemini_keyword_start
            print(f"⏱️ [Timing] Gemini 키워드 생성: {gemini_keyword_time:.3f}초")

        # ⏱️ 2. 알고리즘 로직 (코사인 유사도) 시간 측정
        algorithm_start = time.time()
        recommended_movies = get_content_based_recommendations(search_keywords, start_year, end_year, page)
        algorithm_time = time.time() - algorithm_start
        print(f"⏱️ [Timing] 알고리즘 실행 (코사인 유사도): {algorithm_time:.3f}초")

        # ⏱️ 3. Gemini API (이유/음식 생성) 시간 측정
        gemini_reason_start = time.time()
        if recommended_movies:
            final_movies = ask_gemini_for_reason_and_food(emotions, recommended_movies)
        else:
            final_movies = []
        gemini_reason_time = time.time() - gemini_reason_start
        print(f"⏱️ [Timing] Gemini 이유/음식 생성: {gemini_reason_time:.3f}초")

        # ⏱️ 4. TMDB API 시간 측정 (각 영화마다 호출)
        tmdb_start = time.time()
        response_data = []
        for m in final_movies:
            # 감독 정보 추가 반환
            poster, overview, director = get_tmdb_info(m['title'])
            
            response_data.append({
                "title": m['title'],
                "year": m['year'],
                "genre": m['genre'],
                "score": m['similarity'],
                "poster": poster,
                "overview": overview,
                "director": director,  # 감독 정보 추가
                "reason": m.get('reason', '추천 영화'),
                "food": m.get('food', '맛있는 간식')
            })
        tmdb_time = time.time() - tmdb_start
        print(f"⏱️ [Timing] TMDB API 호출 ({len(final_movies)}개 영화): {tmdb_time:.3f}초")

        # ⏱️ 전체 시간 계산
        total_time = time.time() - total_start
        
        # 📊 종합 통계 출력
        print("\n" + "="*60)
        print("📊 [성능 분석 요약]")
        print("="*60)
        print(f"  1️⃣  Gemini 키워드 생성:     {gemini_keyword_time:>8.3f}초  ({gemini_keyword_time/total_time*100:>5.1f}%)")
        print(f"  2️⃣  알고리즘 실행:          {algorithm_time:>8.3f}초  ({algorithm_time/total_time*100:>5.1f}%)")
        print(f"  3️⃣  Gemini 이유/음식 생성:  {gemini_reason_time:>8.3f}초  ({gemini_reason_time/total_time*100:>5.1f}%)")
        print(f"  4️⃣  TMDB API 호출:         {tmdb_time:>8.3f}초  ({tmdb_time/total_time*100:>5.1f}%)")
        print("-"*60)
        print(f"  🎯  전체 응답 시간:         {total_time:>8.3f}초  (100.0%)")
        print("="*60 + "\n")

        return jsonify({
            "status": "success",
            "results": response_data,
            "used_keywords": search_keywords,
            "page": page,
            # 📊 성능 데이터도 응답에 포함 (프론트엔드에서 활용 가능)
            "performance": {
                "gemini_keyword_time": round(gemini_keyword_time, 3),
                "algorithm_time": round(algorithm_time, 3),
                "gemini_reason_time": round(gemini_reason_time, 3),
                "tmdb_time": round(tmdb_time, 3),
                "total_time": round(total_time, 3)
            }
        })

    except Exception as e:
        total_time = time.time() - total_start
        print(f"❌ [Error] 요청 실패 (소요 시간: {total_time:.3f}초): {e}")
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5001)