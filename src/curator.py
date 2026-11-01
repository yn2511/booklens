"""BookLens AI NLP Curation Engine.
Curriculum Mapping: Week 9 (Statistical NLP with TF-IDF Vectorizer & Cosine Similarity).
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BookLensNLPCurator:
    def __init__(self):
        self.category_meta = {
            "picture_book": {
                "name": "그림책 (Picture Book)",
                "target": "유아 및 초등 저학년 (시각적 상상력과 정서 발달 단계)",
                "theme": "직관적인 일러스트 중심의 서사와 따뜻한 감성 스토리텔링",
                "default_query": "따뜻한 이웃 가족 상상력 마음 공감 일러스트 감동"
            },
            "text_book": {
                "name": "줄글책/문고판 (Text Book)",
                "target": "초등 고학년 및 청소년 (문해력과 사고의 깊이를 넓히는 성장 독서)",
                "theme": "체계적인 문장 서사와 인물 심리 묘사, 주제 의식 탐구",
                "default_query": "어린 시절 성장 우정 생명 용기 자유 문학 감동"
            },
            "comic_book": {
                "name": "만화책/그래픽노블 (Comic Book)",
                "target": "초등 전 학년 및 청소년 (시각적 칸 연출을 통한 빠른 몰입과 흥미 유발)",
                "theme": "다이내믹한 컷 분할과 말풍선 대사, 역동적인 사건 전개",
                "default_query": "역사 문화 모험 과학 원리 만화 스토리텔링 흥미"
            }
        }
        self.books_corpus = [
            {"cat": "picture_book", "title": "달 샤베트", "author": "백희나", "desc": "무더운 여름밤 녹아내리는 달로 샤베트를 만들어 나누는 따뜻한 이웃 간의 연대와 독창적인 입체 일러스트 상상력"},
            {"cat": "picture_book", "title": "구름빵", "author": "백희나", "desc": "나뭇가지에 걸린 구름으로 빵을 구워 먹고 하늘을 날아 아빠의 출근을 돕는 일상 속 판타지와 따뜻한 가족애 상상력"},
            {"cat": "picture_book", "title": "알사탕", "author": "백희나", "desc": "마음의 소리가 들리는 신비한 알사탕을 통해 가족과 친구, 반려견의 진심을 이해하고 공감하는 따뜻한 마음 성장 이야기"},
            {"cat": "picture_book", "title": "수박 수영장", "author": "안녕달", "desc": "커다란 수박 속 시원한 여름 마을 축제와 세대 간의 정겨운 어울림을 청량한 색채 일러스트로 담아낸 그림책"},
            {"cat": "text_book", "title": "나의 라임오렌지나무", "author": "J.M. 바스콘셀로스", "desc": "가난 속에서도 상상력을 잃지 않는 소년 제제가 라임오렌지나무 밍기뉴와 우정을 나누며 아픔을 딛고 자라나는 성장 문학"},
            {"cat": "text_book", "title": "마당을 나온 암탉", "author": "황선미", "desc": "양계장을 벗어나 자유와 꿈을 향해 나아가는 암탉 잎싹이 청둥오리 초록머리를 키워내는 생명의 용기와 모성애 감동 문학"},
            {"cat": "text_book", "title": "샬롯의 거미줄", "author": "E.B. 화이트", "desc": "아기 돼지 윌버와 거미 샬롯의 헌신적인 우정과 생명의 소중함을 따뜻한 문장으로 그려낸 초등 필독 성장 동화"},
            {"cat": "text_book", "title": "아몬드", "author": "손원평", "desc": "감정을 느끼지 못하는 소년이 타인과의 관계와 우정을 통해 공감과 성장의 의미를 배워가는 청소년 문학"},
            {"cat": "comic_book", "title": "먼나라 이웃나라", "author": "이원복", "desc": "세계 각국의 역사와 문화, 사회적 배경을 풍부한 만화 컷 연출과 재치 있는 해설로 쉽게 풀어낸 역사 교양 만화"},
            {"cat": "comic_book", "title": "마법천자문", "author": "올댓스토리", "desc": "손오공의 역동적인 모험 스토리텔링과 한자 마법 연출을 결합하여 어휘력과 흥미를 동시에 높이는 학습 만화"},
            {"cat": "comic_book", "title": "어린이 과학동아 그래픽노블", "author": "동아사이언스", "desc": "우주와 생태계, 물리·화학의 과학 원리를 흥미진진한 탐험 플롯과 만화 연출로 전달하는 과학 그래픽노블"},
            {"cat": "comic_book", "title": "미생", "author": "윤태호", "desc": "바둑의 원리와 사회 초년생의 성장을 세밀한 심리 묘사와 만화적 연출로 그려낸 작품"}
        ]

    def retrieve_similar_books(self, predicted_class, user_interest=None, top_k=3):
        meta = self.category_meta.get(predicted_class, self.category_meta["picture_book"])
        query = user_interest if user_interest else meta["default_query"]
        candidates = [b for b in self.books_corpus if b["cat"] == predicted_class]
        docs = [f"{b['title']} {b['desc']}" for b in candidates]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(docs + [query])
        sims = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
        ranked_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for i in ranked_idx:
            item = dict(candidates[i])
            item["sim_score"] = float(sims[i])
            results.append(item)
        return query, results

    def generate_curation(self, predicted_class, confidence, margin=None, edge_density=None, user_interest=None):
        meta = self.category_meta.get(predicted_class, self.category_meta["picture_book"])
        query, recs = self.retrieve_similar_books(predicted_class, user_interest=user_interest, top_k=3)
        extra_info = ""
        if margin is not None and edge_density is not None:
            guard_note = "안정적 분류 (High Confidence)" if margin >= 0.15 else "경계선 샘플 (Sobel 엣지 밀도 교차 검증 적용)"
            extra_info = f"\n• 판별 보조 지표: Softmax Margin {margin*100:.1f}%p | Sobel 엣지 밀도 {edge_density:.1f}% ({guard_note})"
        report = f"""======================================================================
[BookLens AI 비전 인식 & TF-IDF 기반 맞춤형 큐레이션 리포트]
• 비전 인식 결과: {meta['name']} (인식 신뢰도: {confidence*100:.1f}%){extra_info}
• 권장 독자층: {meta['target']}
• 핵심 테마 및 특징: {meta['theme']}
• NLP 매칭 질의어(Query): "{query}"
----------------------------------------------------------------------
[TF-IDF 코사인 유사도 상위 추천 도서 3선]
1. 《{recs[0]['title']}》 ({recs[0]['author']} 저) — [TF-IDF 유사도: {recs[0]['sim_score']:.3f}]
   └ 줄거리 및 추천 근거: {recs[0]['desc']}
2. 《{recs[1]['title']}》 ({recs[1]['author']} 저) — [TF-IDF 유사도: {recs[1]['sim_score']:.3f}]
   └ 줄거리 및 추천 근거: {recs[1]['desc']}
3. 《{recs[2]['title']}》 ({recs[2]['author']} 저) — [TF-IDF 유사도: {recs[2]['sim_score']:.3f}]
   └ 줄거리 및 추천 근거: {recs[2]['desc']}
======================================================================"""
        return report
