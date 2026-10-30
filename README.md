# 📚 BookLens AI (북렌즈)
> **실시간 도서 표지 인식(전이학습)과 LLM 도서 큐레이션 융합 시스템 및 실환경 영상 왜곡 극복**  
> *가천대학교 IT융합대학 컴퓨터공학과 기계학습프로그래밍 최종 프로젝트 (지도교수: 조윤실 교수님)*  
> *기획 및 개발: 윤나래 (202533791)*

[![Deploy with Cloudflare Pages](https://img.shields.io/badge/Deploy-Cloudflare_Pages-orange?logo=cloudflare)](https://pages.cloudflare.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)

---

## 📌 1. 프로젝트 개요 (Executive Summary)
**BookLens AI**는 기계학습프로그래밍 과제 3의 **초등학교 도서관 도서 형태 분류(`book_dataset`: 그림책, 줄글책, 만화책)** 모델을 직접 계승하여, 실시간 카메라 영상으로 책을 비췄을 때 책의 형태를 판별하고 **LLM이 권장 독자층, 핵심 줄거리 테마, 유사 도서 3권을 실시간 큐레이션**해 주는 멀티모달 북 도슨트 서비스입니다.

- **과제 3의 연속성**: 과제 3에서 구축한 3개 클래스 전이학습 모델과 검증된 데이터셋을 100% 활용.
- **핵심 난관 극복 (Engineering Value)**: 실시간 스마트폰 카메라 영상의 조명 반사, 각도 원근 왜곡(Perspective) 문제를 `RandomPerspective`와 `ColorJitter` 복합 데이터 증강으로 극복하여 95%+ 강건성 확보.
- **인프라 & 배포**: Cloudflare Pages 글로벌 엣지 네트워크 배포로 서버 비용 0원, 도메인 1회성 결제로 영구 무료 실서비스 완성.

---

## 🏗️ 2. 시스템 및 모델 아키텍처 (Architecture)

```text
               [ Real-time Camera / Book Cover Image ]
                                 │
             ┌───────────────────┴───────────────────┐
             │ Pretrained ResNet-18 Feature Extractor│ (ImageNet Weights)
             │ Trained with Robust Augmentation      │ (Perspective + ColorJitter)
             └───────────────────┬───────────────────┘
                                 │
             [ Vision Prediction & Softmax Confidence ]
             • picture_book (97.4%) / text_book / comic_book
                                 │
             ┌───────────────────┴───────────────────┐
             ▼                                       ▼
    [ Structured Prompt Engine ]           [ Cloudflare Pages Web ]
    • Target Reader Demographics           • In-Browser Instant UI
    • Story Theme & Key Appeal             • Zero-Server Architecture
    • 3 Curated Similar Books
```

---

## 📊 3. 난관 극복 튜닝 실험 결과 (Ablation Study)

| 학습 방식 및 난관 극복 기법 | 학습 파라미터 수 | 학습 소요시간 | Test Accuracy | Test Macro-F1 |
|---|---|---|---|---|
| 1. Scratch (small_cnn) | 277,411개 | 14.2초 | 68.89% | 0.6721 |
| 2. Feature Extractor (ResNet18 동결) | 1,539개 | 8.6초 | 87.78% | 0.8654 |
| 3. Fine-Tuning (기본 증강) | 11,178,051개 | 19.4초 | 92.22% | 0.9180 |
| **4. [난관 극복] Fine-Tuning + Robust Aug + CosineLR** | **11,178,051개** | **20.1초** | **96.67%** | **0.9655** |

> **분석 요약**: 실제 도서관 및 조명 환경을 반영한 `RandomPerspective` 및 `ColorJitter` 복합 증강을 적용한 4번 모델이 원근 왜곡 및 반사광 테스트 환경에서도 96.67%의 압도적인 정확도와 최고 F1-score를 기록함.

---

## 🚀 4. 프로젝트 산출물
- [BookLens 공식 PRD 기획서 다운로드](docs/BookLens_PRD.docx)
- [Colab 학습 및 튜닝 파이프라인](model/기계학습_최종프로젝트_BookLens_202533791_윤나래.ipynb)
- [실시간 인터랙티브 웹 프론트엔드](index.html)
