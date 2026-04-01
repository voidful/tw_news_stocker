#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重新處理所有新聞資料，使用ML模型計算sentiment分數
Reprocess all news data with ML-based sentiment analysis
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path to import from fetch_and_rank.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fetch_and_rank import (
    load_company_alias,
    load_sentiment_model,
    load_sentiment_lexicon,
    score_sentence_ml,
    score_sentence,
    split_sentences,
    normalize_for_match,
    cap_scores,
    DOCS_DATA,
    NEWS_DIR,
    SENT_SPLIT,
    MAX_PER_ARTICLE_ABS,
    DEFAULT_POS,
    DEFAULT_NEG
)

def reprocess_news_file(filepath: Path, tokenizer, model, alias_to_codes, senti_kp):
    """
    重新處理單個新聞JSON檔案
    Reprocess a single news JSON file with ML sentiment
    """
    print(f"Processing: {filepath.name}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            news_list = json.load(f)
    except Exception as e:
        print(f"  Error loading file: {e}")
        return 0
    
    updated_count = 0
    comp_kp = {}
    
    for news_item in news_list:
        # Get title and extract text
        title = news_item.get('title', '')
        # Combine title for full text (summary might not be stored)
        text = normalize_for_match(title)
        
        # Split into sentences
        sent_list = split_sentences(text, SENT_SPLIT)
        
        # Recompute sentiment scores
        per_company = {}
        details_all = []
        art_pos = 0.0
        art_neg = 0.0
        
        for sent in sent_list:
            # Use ML-based scoring if model is available
            if tokenizer and model:
                sc, det = score_sentence_ml(sent, comp_kp, alias_to_codes, tokenizer, model, src_w=1.0)
            else:
                sc, det = score_sentence(sent, comp_kp, alias_to_codes, senti_kp, src_w=1.0)
            
            # Aggregate company scores
            for k, v in sc.items():
                per_company[k] = per_company.get(k, 0.0) + float(v)
            
            # Accumulate sentence details
            det = det or []
            for d in det:
                details_all.append(d)
                val = d.get("final", 0.0)
                if val > 0:
                    art_pos += val
                else:
                    art_neg += abs(val)
        
        # Cap per-company scores
        per_company = cap_scores(per_company, MAX_PER_ARTICLE_ABS)
        
        # Update news item
        news_item['per_company'] = per_company
        news_item['codes'] = list(per_company.keys())
        news_item['sent_pos'] = round(art_pos, 4)
        news_item['sent_neg'] = round(art_neg, 4)
        news_item['sent_score'] = round(art_pos - art_neg, 4)
        news_item['detail'] = details_all
        
        if per_company:
            updated_count += 1
    
    # Write back updated data
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(news_list, f, ensure_ascii=False, indent=2)
        print(f"  Updated {updated_count}/{len(news_list)} articles with company mentions")
    except Exception as e:
        print(f"  Error writing file: {e}")
        return 0
    
    return updated_count

def main():
    print("=" * 60)
    print("重新處理所有新聞資料 - ML Sentiment Analysis")
    print("Reprocessing all news data with ML sentiment model")
    print("=" * 60)
    
    # Load company aliases
    print("\n1. Loading company aliases...")
    alias_to_codes = load_company_alias()
    print(f"   Loaded {len(alias_to_codes)} company aliases")
    
    # Load sentiment lexicon (for fallback)
    print("\n2. Loading sentiment lexicon...")
    pos_lex, neg_lex = load_sentiment_lexicon()
    senti_kp = {
        "pos": [normalize_for_match(w) for w in pos_lex],
        "neg": [normalize_for_match(w) for w in neg_lex]
    }
    print(f"   Loaded {len(pos_lex)} positive and {len(neg_lex)} negative keywords")
    
    # Load ML model
    print("\n3. Loading ML sentiment model...")
    tokenizer, model = load_sentiment_model()
    if tokenizer and model:
        print("   ✓ ML model loaded successfully")
    else:
        print("   ⚠ ML model not available, using keyword-based fallback")
    
    # Find all news JSON files
    print("\n4. Finding news files...")
    news_files = sorted(NEWS_DIR.glob("*.json"))
    print(f"   Found {len(news_files)} news files to process")
    
    if not news_files:
        print("\n   No news files found in:", NEWS_DIR)
        return
    
    # Process each file
    print("\n5. Processing news files...\n")
    total_updated = 0
    
    for filepath in news_files:
        count = reprocess_news_file(filepath, tokenizer, model, alias_to_codes, senti_kp)
        total_updated += count
    
    print("\n" + "=" * 60)
    print(f"✓ Complete! Updated {total_updated} articles across {len(news_files)} files")
    print("=" * 60)

if __name__ == "__main__":
    main()
