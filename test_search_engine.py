"""
test_search_engine.py
SearchEngine 테스트 스크립트
"""

import sys
import json
from pathlib import Path
from src.s6_search_engine import SearchEngine
from src.s5_embedding_manager import EmbeddingManager
import faiss
import os
from dotenv import load_dotenv

load_dotenv()

def load_components(institution="kb"):
    """저장된 컴포넌트들 로드"""
    print(f"📂 {institution.upper()} 컴포넌트 로딩 중...\n")
    
    base_path = Path(f"data/vector_store/{institution}")
    
    # FAISS 인덱스 로드
    faiss_path = base_path / "faiss_index.bin"
    faiss_index = faiss.read_index(str(faiss_path))
    print(f"✓ FAISS 인덱스 로드: {faiss_index.ntotal}개 벡터")
    
    # 메타데이터 로드
    metadata_path = base_path / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"✓ 메타데이터 로드: {len(metadata)}개")
    
    # 청크 데이터 로드 (processed에서)
    chunks_path = Path(f"data/processed/{institution}/{institution}_chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"✓ 청크 데이터 로드: {len(chunks)}개\n")
    
    return faiss_index, metadata, chunks


def test_search_engine(institution="kb"):
    """SearchEngine 테스트"""
    
    # 1. 컴포넌트 로드
    faiss_index, metadata, chunks = load_components(institution)

    print("🔧 EmbeddingManager 초기화...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 .env 파일에 없습니다!")
    
    embedding_manager = EmbeddingManager(openai_api_key=api_key)
    print()
       
    # 3. SearchEngine 초기화
    print("🔧 SearchEngine 초기화...")
    search_engine = SearchEngine(
        faiss_index=faiss_index,
        metadata=metadata,
        chunks=chunks,
        embedding_manager=embedding_manager
    )
    print()
    
    # 4. 테스트 쿼리들
    test_queries = [
        "2024년 부동산 시장 전망은?",
        "서울 아파트 가격 동향",
        "금리 인상이 부동산에 미치는 영향"
    ]
    
    print("="*80)
    print(f"🔍 {institution.upper()} 검색 테스트 시작")
    print("="*80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[쿼리 {i}] {query}")
        print("-"*80)
        
        # 벡터 검색
        print("\n📊 벡터 검색 결과 (Top 3):")
        vector_results = search_engine.vector_search(query, top_k=3)
        for result in vector_results:
            print(f"  {result['rank']}. [점수: {result['score']:.3f}]")
            print(f"     {result['content'][:100]}...")
            print()
        
        # 키워드 검색
        print("🔤 키워드 검색 결과 (Top 3):")
        keyword_results = search_engine.keyword_search(query, top_k=3)
        for result in keyword_results:
            print(f"  {result['rank']}. [점수: {result['score']:.3f}]")
            print(f"     {result['content'][:100]}...")
            print()
        
        # 하이브리드 검색
        print("🎯 하이브리드 검색 결과 (Top 5):")
        hybrid_results = search_engine.hybrid_search(query, top_k=5)
        for result in hybrid_results:
            print(f"  {result['rank']}. [RRF 점수: {result['rrf_score']:.4f}]")
            print(f"     {result['content'][:100]}...")
            print(f"     출처: {result['metadata'].get('source', 'Unknown')}")
            print()
    
    print("="*80)
    print("✅ 테스트 완료!")
    print("="*80)


def interactive_mode(institution="kb"):
    """대화형 검색 모드"""
    
    # 컴포넌트 로드
    faiss_index, metadata, chunks = load_components(institution)
    
    # 초기화
    embedding_manager = EmbeddingManager()
    search_engine = SearchEngine(
        faiss_index=faiss_index,
        metadata=metadata,
        chunks=chunks,
        embedding_manager=embedding_manager
    )
    
    print("\n" + "="*80)
    print(f"🔍 {institution.upper()} 대화형 검색 모드")
    print("="*80)
    print("검색어를 입력하세요 (종료: 'q' 또는 'exit')")
    print("-"*80)
    
    while True:
        query = input("\n💬 검색어: ").strip()
        
        if query.lower() in ['q', 'quit', 'exit']:
            print("\n👋 검색을 종료합니다.")
            break
        
        if not query:
            continue
        
        print(f"\n🔎 '{query}' 검색 중...\n")
        
        # 하이브리드 검색
        results = search_engine.hybrid_search(query, top_k=5)
        
        if not results:
            print("❌ 검색 결과가 없습니다.")
            continue
        
        print(f"📋 검색 결과 ({len(results)}개):")
        print("-"*80)
        
        for result in results:
            print(f"\n[{result['rank']}] RRF 점수: {result['rrf_score']:.4f}")
            print(f"출처: {result['metadata'].get('source', 'Unknown')}")
            print(f"페이지: {result['metadata'].get('page', 'N/A')}")
            print(f"내용: {result['content'][:200]}...")
            print("-"*80)


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         SearchEngine 테스트 스크립트                      ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # 기관 선택
    print("테스트할 기관을 선택하세요:")
    print("1. KB (kb)")
    print("2. HD (hd)")
    print("3. KHI (khi)")
    
    inst_choice = input("\n기관 선택 (1/2/3): ").strip()
    institution_map = {"1": "kb", "2": "hd", "3": "khi"}
    institution = institution_map.get(inst_choice, "kb")
    
    print(f"\n선택: {institution.upper()}\n")
    
    # 모드 선택
    print("모드를 선택하세요:")
    print("1. 자동 테스트 (미리 정의된 쿼리)")
    print("2. 대화형 검색")
    
    mode_choice = input("\n선택 (1 또는 2): ").strip()
    
    if mode_choice == "1":
        test_search_engine(institution)
    elif mode_choice == "2":
        interactive_mode(institution)
    else:
        print("❌ 잘못된 선택입니다. 1 또는 2를 입력하세요.")