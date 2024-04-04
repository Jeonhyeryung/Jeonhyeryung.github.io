---

title:  "[Study] GeoNeRF : Overview of RAG"

excerpt: "패스트 캠퍼스 강의 정리"
categories:
  - NLP
toc_label: "  목차"
toc: true
toc_sticky: true

date: 2024-03-28
last_modified_at: 2024-03-28
---
<span style="color:gray;">Reference 1: "RAG를 활용한 완성도 높은 LLM 서비스 구축 With langchain & llamaindex." 패스트 캠퍼스</span>

<span style="color:gray;">Reference 2: Yunfan et. al Retrieval-Augmented Generation for Large Language Models: A Survey</span>



## RAG Pipeline

---

<p style="text-align: center;">
  <img src="/images/rag_1.png" width="80%">
</p>

- 기존 LLM 에 Indexing과 Retreival가 추가

### Indexing

- 문서를 관리하기 쉬운 단위로 나누는 것을 chunking이라고 함
- 각 chunk에 해당하는 embedding을 미리 만들어 둠

### Retrieval

- 인덱싱 과정에서 잘 chunking해놓은 문서 사이에서 사용자의 쿼리에 가장 도움이 될 수 있는 문서를 찾아내는 것
- 가장 흔한 방법은 사용자의 쿼리와 가장 유사한 문서를 찾는 것
- 유사도가 높은 여러 chunk를 참고하여 답변을 생성하도록 할 수 있음 (정보가 여러 chunk에 적재되어 있을 수 있음)

### Generation

- 유저의 쿼리와 retrieval된 문서를 합쳐서 사용자에게 전달할 응답을 생성

## Static RAG

---

<p style="text-align: center;">
  <img src="/images/rag_2.png" width="70%">
</p>

### Retreival

- **TF-IDF:** 문서에서 단어의 상대적 중요도를 파악
- **BM25:** TF-IDF와 비슷하지만 문서의 길이와 같은 다른 요소를 좀 더 세심하게 고려해서 관련성을 측정

### Generation

- fine-tuning 없이 사전 학습된 LLM을 그대로 가져다 쓰는 것

## Naive RAG, Advanced RAG, Modular RAG

---

### Naive RAG

- 이전 RAG 구조와 동일 (간단!)
- **한계점**
    - Retrieval 정확도가 낮거나, Retrieval하고자 하는 내용이 이미 outdated 되어 있을 수 있음
    - LLM을 이용하기 때문에 hallucination이나 bias와 같은 문제에서 완전히 자유로울 수 없음
    - 문서를 Generate해서 전달하는 방식에 문제가 있을 수도 있음 → 현재 task와 잘 align되지 않은 방향으로 전달을 해줄 수도 있고, retreival된 정보들 간 중복이나 노이즈를 제대로 필터링해주지 못해서 성능이 저하될 수도 있음
    - 혹은 Generation 과정이 retrieval된 문서에 과도하게 의존해서 검색엔진 이상의 가치를 제공하지 못할 수도 있음

### Advanced RAG

- Naive RAG 구조 자체는 유지하면서 각각의 단계를 조금씩 보완하고 전처리, 후처리를 추가
- **Pre-retrieval:** 데이터 인덱싱 단계를 최적화
    - 데이터 품질 향상 (관련성이 적은 정보 삭제, 데이터 최신성 확보)
    - 인덱스 구조 최적화 (chunk 사이즈 조절, 여러 인덱스 사이에서 질의)
    - 메타데이터 추가 (일자 등)
    - 데이터 사이의 alignment 향상 (SANTA)
- **Retrieval:**
    - 임베딩 모델을 data specific하게 fine tuning
    - 다이나믹 임베딩 사용
- **Post-Retrieval:**
    - **Re-Ranking:** 문서들 간 ranking
        - **`DiversityRanker`**
            1. Sentence transformer 모델로 임베딩 생성
            2. 쿼리와 가장 가까운 문서 선택
            3. 이미 선택된 문서들과 다른 문서의 거리 계산  
            4. 선택된 문서들과 평균적으로 가장 먼 문서 선택 
        - **`LostIntheMiddleRanker`**
            - 가장 정확한 문서를 prompt의 처음과 끝에 위치하게 함
        - **`Relevance Score`**
            - cohere AI, bge-rerank: query+document를 LLM 인풋으로넣어 relevance score 계산
            - LongLLMLingua: 각각의 다큐먼트에 쿼리를 컨디셔닝한 perplexity계산
    - **Prompt Compression**
        - **Compressor** 활용: 쓸모있는 정보만 남겨보자!
            - **`RECOMP:`**
                - Extractive Compressor: input 쿼리와 document 안에 포함된 각각의 sentence의 embedding distance로 점수를 구함 → 유사도가 높을수록 유용한 문장이라고 보고, 그런 문장들만 prompt에 넣어줌
                - Abstractive Compressor: 인코더 디코더 모듈에 document를 넣어서 요약 문서를 생성
            - **`MemWalker:`**
                1. Memory Tree Construction
                2. Navigation

### Modular RAG

- retrieve 뿐만 아니라 search, rewrite 등 새로운 여러 모듈의 등장
- 이런 모듈들을 조합해서 RAG 프로세스 생성하면 모두 Modular RAG의 일종

<p style="text-align: center;">
  <img src="/images/rag_3.png" width="70%">
</p>

- **New modules:**
    - **`SearchModule`**
        - 쿼리와 document의 embedding 서치 이외에 추가적인 검색 시나리오를 가능하게 함
        - KnowledGPT: PoT 방식으로 LLM이 쿼리 생성
    - **`Memory module`**
        - LLM이 생성한 텍스트를 메모리로 활용
        - Selfmem:
            1. 데이터셋으로부터 메모리를 리트리브
            2. Generator가 메모리를 사용해 candidate pool 생성
            3. selector가 candidate 중에서 memory를 선택 
    - **`Fusion`**
        - LLM이 유저 쿼리로부터 여러 개의 multi-query 생성해서 활용
    - **`Routing`**
        - LLM 콜을 통해 유저 쿼리에 따라 후속 행동을 결정
        - 어떤 데이터 스토어에 접속할 것인지, 혹은 요약할 것인지 등

## RAG and the limitations of LLMS

---

- **Hallucination**
    - 모델이 정확하지 않은 정보를 마치 정확한 것처럼 전달하는 현상
    - RAG 프레임워크를 사용할 경우 retriever가 데이터소스에서 쿼리와 유사도가 높은 문서를 retreiver 해 옴
    - language 모델은 이 문서를 받아서 생성을 하게 되기 때문에 잘못된 정보를 전달할 가능성이 훨씬 줄어들게 됨

- **Outdated Knowledge**
    - llm이 짧은 주기로 학습하기에는 비용이 너무 비싸다는 문제에서 기인
        - 대부분은 foundation 모델을 가져와서 사용하는데, 이 경우 주기적인 업데이트는 더 어려움
    - RAG 프레임워크를 사용할 경우 data source의 데이터만 업데이트하면 되기 때문에 쉽게 이 문제를 우회할 수 있음
        - preprocess 과정과 embedding 과정만 다시 해주면 됨

- **Untraceable reasoning process**
    - 추론 과정이 불투명함
    - RAG 프레임워크를 사용할 경우 추론이 어디부터 잘못 되었는지 (명확하게는 아니지만) 확인 가능
        - retreiver이 잘못 되었는지
        - retreiver은 잘 되었지만 generation 단계에서 문서가 반영이 잘 안되고 있는지

- **Bias**
    - RAG 프레임워크가 해결하지 못한 문제점으로, 모델이 편향된 정보를 생성하는 것을 완전히 막기는 어려움
    - 또한 수집되는 데이터 자체가 편향성을 지니고 있을 수도 있음
