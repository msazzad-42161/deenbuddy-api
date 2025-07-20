# main.py - Enhanced for Quran + Tafseer Support
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
from typing import List, Dict
import re

# Load environment variables
load_dotenv()

app = FastAPI(title="Islamic Knowledge API - Complete Quran + Tafseer Edition")

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./islamic_knowledge_db")

# Use sentence transformers for embeddings
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create or get collection
collection = chroma_client.get_or_create_collection(
    name="islamic_texts",
    embedding_function=sentence_transformer_ef
)

class QuestionRequest(BaseModel):
    question: str
    max_results: int = 6  # Increased for verse + tafseer
    include_tafseer: bool = True

class IslamicResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]
    greeting: str = "Bismillah, Akhi!"
    total_texts_searched: int
    verses_found: int = 0
    tafseer_found: int = 0

def search_islamic_knowledge(question: str, n_results: int = 6, include_tafseer: bool = True):
    """Enhanced search for Quran verses and tafseer"""
    try:
        # Search for relevant texts
        results = collection.query(
            query_texts=[question],
            n_results=n_results
        )
        
        relevant_texts = []
        verse_count = 0
        tafseer_count = 0
        
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            relevance = (1 - distance) * 100
            
            category = metadata.get("category", "Unknown")
            
            # Count categories
            if category == "Quran":
                verse_count += 1
            elif category == "Tafseer":
                tafseer_count += 1
                if not include_tafseer:
                    continue  # Skip tafseer if not requested
            
            relevant_texts.append({
                "text": doc,
                "source": metadata.get("source", "Unknown"),
                "category": category,
                "surah_name": metadata.get("surah_english_name", ""),
                "full_reference": metadata.get("full_reference", metadata.get("source", "")),
                "revelation_type": metadata.get("revelation_type", ""),
                "tafseer_name": metadata.get("tafseer_name", ""),
                "relevance": f"{relevance:.1f}%",
                "distance": distance
            })
        
        # Sort by relevance (distance), but prioritize Quran verses
        relevant_texts.sort(key=lambda x: (x["category"] != "Quran", x["distance"]))
        
        return relevant_texts, verse_count, tafseer_count
        
    except Exception as e:
        print(f"Error searching knowledge base: {e}")
        return [], 0, 0

def generate_enhanced_response_with_tafseer(question: str, relevant_texts: List[Dict], verse_count: int, tafseer_count: int) -> str:
    """Generate comprehensive response using both Quran and Tafseer"""
    
    if not relevant_texts:
        return "Akhi, I couldn't find specific guidance for your question. This might be something to discuss with a local scholar or imam. Wallahu a'lam."
    
    # Separate verses and tafseer
    verses = [t for t in relevant_texts if t["category"] == "Quran"]
    tafseers = [t for t in relevant_texts if t["category"] == "Tafseer"]
    
    response_parts = ["Bismillah, Akhi!"]
    
    # Main answer starts with most relevant content
    main_content = relevant_texts[0]
    
    if main_content["category"] == "Quran":
        response_parts.append(f"\nRegarding your question about \"{question}\", Allah (SWT) provides clear guidance in the Quran.")
        response_parts.append(f"\nIn {main_content['full_reference']}, Allah (SWT) says:")
        
        # Quote the verse (truncate if very long)
        verse_text = main_content['text']
        if len(verse_text) > 250:
            verse_text = verse_text[:250] + "..."
        
        response_parts.append(f"\n\"{verse_text}\"")
        response_parts.append(f"— {main_content['full_reference']}")
        
        # Add tafseer explanation if available
        relevant_tafseer = None
        for tafseer in tafseers:
            # Check if tafseer is for the same verse
            if (tafseer.get('surah_name') == main_content.get('surah_name') and 
                tafseer.get('full_reference', '').split('(')[1].split(')')[0] == 
                main_content.get('full_reference', '').split('(')[1].split(')')[0]):
                relevant_tafseer = tafseer
                break
        
        if relevant_tafseer:
            response_parts.append(f"\n**Scholarly Commentary ({relevant_tafseer['tafseer_name']}):**")
            tafseer_text = relevant_tafseer['text']
            if len(tafseer_text) > 300:
                tafseer_text = tafseer_text[:300] + "..."
            response_parts.append(f"{tafseer_text}")
        
        # Add supporting verse if highly relevant
        if len(verses) > 1 and verses[1]['distance'] < 0.4:
            second_verse = verses[1]
            response_parts.append(f"\nThis is further supported in {second_verse['full_reference']} where Allah also mentions similar guidance.")
    
    else:  # Main content is tafseer
        response_parts.append(f"\nRegarding \"{question}\", Islamic scholars provide valuable insights.")
        response_parts.append(f"\nIn the commentary on {main_content['full_reference']}:")
        
        tafseer_text = main_content['text']
        if len(tafseer_text) > 300:
            tafseer_text = tafseer_text[:300] + "..."
        
        response_parts.append(f"\n{tafseer_text}")
        response_parts.append(f"— {main_content['tafseer_name']}")
    
    # Add contextual Islamic guidance
    add_contextual_guidance(response_parts, question.lower())
    
    # Summary of sources
    if verse_count > 0 and tafseer_count > 0:
        response_parts.append(f"\n*This response draws from {verse_count} Quranic verse(s) and {tafseer_count} scholarly commentary/ies.*")
    elif verse_count > 0:
        response_parts.append(f"\n*This response is based on {verse_count} Quranic verse(s).*")
    elif tafseer_count > 0:
        response_parts.append(f"\n*This response draws from {tafseer_count} scholarly commentary/ies.*")
    
    response_parts.append("\nWallahu a'lam - Allah knows best.")
    response_parts.append("May Allah grant us understanding and guide us to implement His teachings. Barakallahu feek for seeking knowledge!")
    
    return "\n".join(response_parts)

def add_contextual_guidance(response_parts: List[str], question_lower: str):
    """Add contextual Islamic guidance based on question topic"""
    
    guidance_map = {
        'patience': "SubhanAllah, sabr (patience) is one of the most beautiful qualities a believer can develop. It's not passive waiting, but active trust in Allah's timing and wisdom.",
        'sabr': "The Quran mentions sabr over 100 times, showing how central patience is to the Islamic way of life.",
        'prayer': "MashaAllah, salah is indeed the pillar that connects us directly with our Creator, five times a day, keeping us grounded in remembrance of Allah.",
        'salah': "Prayer is the first thing we'll be asked about on the Day of Judgment - may Allah make us among those who establish it properly.",
        'charity': "Zakat and charity purify our wealth and souls. The Quran consistently pairs charity with prayer, showing their equal importance.",
        'zakat': "Giving charity is not just about helping others, but purifying our hearts from the love of dunya.",
        'forgiveness': "Allah's mercy and forgiveness are infinite. Even when we fall short, His door of repentance remains open.",
        'tawbah': "True repentance involves recognizing the sin, feeling remorse, seeking Allah's forgiveness, and resolving not to repeat it.",
        'guidance': "The Quran itself is called 'Al-Huda' (The Guidance) - it's our roadmap for this life and the next.",
        'hidayah': "True guidance comes only from Allah, and it's the greatest blessing one can receive.",
        'parent': "The Quran places obedience to parents right after worship of Allah - showing their elevated status in Islam.",
        'mother': "Paradise lies at the feet of the mother - showing the immense respect Islam has for mothers.",
        'father': "The father is the qawwam (protector/maintainer) of the family, with great responsibility and rights.",
        'marriage': "Marriage in Islam is half of one's deen, completing the spiritual and worldly aspects of life.",
        'spouse': "The relationship between spouses should be based on mutual respect, love, and helping each other grow closer to Allah."
    }
    
    for keyword, guidance in guidance_map.items():
        if keyword in question_lower:
            response_parts.append(f"\n{guidance}")
            break

@app.on_event("startup")
async def startup_event():
    """Check database status on startup"""
    count = collection.count()
    
    # Sample to check what we have
    if count > 0:
        sample = collection.query(query_texts=["Allah"], n_results=min(10, count))
        verse_count = sum(1 for meta in sample['metadatas'][0] if meta.get('category') == 'Quran')
        tafseer_count = sum(1 for meta in sample['metadatas'][0] if meta.get('category') == 'Tafseer')
        
        print(f"✅ Alhamdulillah! Islamic knowledge database ready:")
        print(f"   Total entries: {count}")
        print(f"   Estimated verses: ~{verse_count * (count // 10)}")
        print(f"   Estimated tafseer entries: ~{tafseer_count * (count // 10)}")
    else:
        print("⚠️  Database is empty. Run quran_loader.py and tafseer_loader.py")

@app.get("/")
async def root():
    count = collection.count()
    
    # Quick sample to show what we have
    status_msg = "Ready to serve with Quran"
    try:
        if count > 0:
            sample = collection.query(query_texts=["guidance"], n_results=min(5, count))
            has_tafseer = any(meta.get('category') == 'Tafseer' for meta in sample['metadatas'][0])
            if has_tafseer:
                status_msg += " + Tafseer"
    except:
        pass
    
    return {
        "message": "Islamic Knowledge API - Complete Quran + Tafseer Edition", 
        "status": status_msg,
        "total_entries": count,
        "blessing": "Bismillah! Ready to serve the Ummah with comprehensive Islamic knowledge 🕌📚"
    }

@app.post("/ask", response_model=IslamicResponse)
async def ask_islamic_question(request: QuestionRequest):
    """Ask an Islamic question and get comprehensive response from Quran + Tafseer"""
    try:
        # Search for relevant Islamic texts
        relevant_texts, verse_count, tafseer_count = search_islamic_knowledge(
            request.question, 
            request.max_results,
            request.include_tafseer
        )
        
        if not relevant_texts:
            return IslamicResponse(
                answer="Akhi, I couldn't find specific guidance for your question in my current knowledge base. This might be something to discuss with a local scholar. Wallahu a'lam.",
                sources=[],
                greeting="Bismillah, Akhi!",
                total_texts_searched=collection.count(),
                verses_found=0,
                tafseer_found=0
            )
        
        # Generate comprehensive response
        answer = generate_enhanced_response_with_tafseer(
            request.question, 
            relevant_texts,
            verse_count,
            tafseer_count
        )
        
        # Format sources with complete information
        sources = []
        for text in relevant_texts[:5]:  # Limit sources in response
            source_info = {
                "reference": text["full_reference"],
                "category": text["category"],
                "relevance": text["relevance"],
                "surah_name": text.get("surah_name", ""),
                "revelation_type": text.get("revelation_type", ""),
                "tafseer_name": text.get("tafseer_name", "") if text["category"] == "Tafseer" else ""
            }
            sources.append(source_info)
        
        return IslamicResponse(
            answer=answer,
            sources=sources,
            greeting="Bismillah, Akhi!",
            total_texts_searched=collection.count(),
            verses_found=verse_count,
            tafseer_found=tafseer_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/stats")
async def database_stats():
    """Get comprehensive statistics about the Islamic knowledge database"""
    try:
        total_count = collection.count()
        
        if total_count == 0:
            return {"message": "Database is empty. Run loaders first."}
        
        # Comprehensive sampling for accurate stats
        sample_size = min(200, total_count)
        sample_results = collection.query(
            query_texts=["Allah guidance"],
            n_results=sample_size
        )
        
        categories = {}
        revelation_types = {}
        tafseers = {}
        surahs = {}
        
        for metadata in sample_results['metadatas'][0]:
            # Categories
            cat = metadata.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
            
            # Revelation types (only for Quran)
            if cat == 'Quran':
                rev_type = metadata.get('revelation_type', 'Unknown')
                if rev_type != 'Unknown':
                    revelation_types[rev_type] = revelation_types.get(rev_type, 0) + 1
                
                # Surah distribution
                surah_name = metadata.get('surah_english_name', 'Unknown')
                if surah_name != 'Unknown':
                    surahs[surah_name] = surahs.get(surah_name, 0) + 1
            
            # Tafseer distribution
            elif cat == 'Tafseer':
                tafseer_name = metadata.get('tafseer_name', 'Unknown')
                if tafseer_name != 'Unknown':
                    tafseers[tafseer_name] = tafseers.get(tafseer_name, 0) + 1
        
        # Calculate estimates based on sample
        scale_factor = total_count / sample_size if sample_size > 0 else 1
        
        estimated_stats = {
            "categories": {k: int(v * scale_factor) for k, v in categories.items()},
            "revelation_types": {k: int(v * scale_factor) for k, v in revelation_types.items()},
            "available_tafseers": list(tafseers.keys()),
            "sample_surahs": list(surahs.keys())[:10]  # Show first 10 surahs found
        }
        
        return {
            "total_entries": total_count,
            "estimated_breakdown": estimated_stats,
            "status": "Alhamdulillah - Comprehensive Islamic knowledge base ready!",
            "capabilities": [
                "Complete Quran verses with context",
                "Multiple tafseer commentaries",
                "Scholarly interpretations",
                "Cross-referenced responses"
            ],
            "sample_questions": [
                "What does the Quran say about patience during hardship?",
                "How should I treat my parents according to Islam?",
                "What is the importance of prayer in a believer's life?",
                "Tell me about charity and helping the poor in Islam"
            ]
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/search/{query}")
async def quick_search(query: str, limit: int = 8, category: str = "all"):
    """Enhanced quick search with category filtering"""
    try:
        results, verse_count, tafseer_count = search_islamic_knowledge(query, limit, True)
        
        # Filter by category if specified
        if category.lower() != "all":
            if category.lower() == "quran":
                results = [r for r in results if r["category"] == "Quran"]
            elif category.lower() == "tafseer":
                results = [r for r in results if r["category"] == "Tafseer"]
        
        formatted_results = []
        for r in results:
            preview_length = 150 if r["category"] == "Tafseer" else 100
            text_preview = r["text"][:preview_length] + "..." if len(r["text"]) > preview_length else r["text"]
            
            result_item = {
                "category": r["category"],
                "reference": r["full_reference"],
                "text_preview": text_preview,
                "relevance": r["relevance"],
                "surah": r.get("surah_name", "")
            }
            
            if r["category"] == "Tafseer":
                result_item["tafseer_source"] = r.get("tafseer_name", "")
            
            formatted_results.append(result_item)
        
        return {
            "query": query,
            "category_filter": category,
            "results_found": len(results),
            "verses_found": verse_count,
            "tafseer_found": tafseer_count,
            "results": formatted_results
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """Enhanced health check with detailed status"""
    try:
        count = collection.count()
        
        # Quick capability check
        capabilities = {
            "database_accessible": True,
            "has_content": count > 0,
            "quran_loaded": False,
            "tafseer_loaded": False
        }
        
        if count > 0:
            # Sample to check capabilities
            sample = collection.query(query_texts=["Allah"], n_results=min(10, count))
            
            for metadata in sample['metadatas'][0]:
                if metadata.get('category') == 'Quran':
                    capabilities["quran_loaded"] = True
                elif metadata.get('category') == 'Tafseer':
                    capabilities["tafseer_loaded"] = True
        
        health_status = "healthy" if capabilities["has_content"] else "needs_data"
        
        return {
            "status": health_status,
            "database_entries": count,
            "capabilities": capabilities,
            "message": "Alhamdulillah, comprehensive Islamic knowledge system is running!",
            "recommendations": [
                "Run quran_loader.py if no Quran verses detected",
                "Run tafseer_loader.py to add scholarly commentaries",
                "Use /ask endpoint for comprehensive Islamic guidance"
            ] if not capabilities["quran_loaded"] else [
                "System fully operational - ready for Islamic Q&A!",
                "Try asking complex questions for verse + tafseer responses"
            ]
        }
        
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/surah/{surah_number}")
async def get_surah_info(surah_number: int):
    """Get information about a specific surah"""
    try:
        # Search for verses from this surah
        results = collection.query(
            query_texts=["Allah guidance"],  # Generic query
            n_results=200,  # Get many results
            where={"surah_number": str(surah_number)}  # Filter by surah
        )
        
        if not results['documents'][0]:
            return {"error": f"No data found for Surah {surah_number}"}
        
        # Get surah info from first result
        first_metadata = results['metadatas'][0][0]
        surah_info = {
            "surah_number": surah_number,
            "arabic_name": first_metadata.get('surah_name', 'Unknown'),
            "english_name": first_metadata.get('surah_english_name', 'Unknown'),
            "revelation_type": first_metadata.get('revelation_type', 'Unknown'),
        }
        
        # Count verses and tafseer entries
        verse_count = sum(1 for meta in results['metadatas'][0] if meta.get('category') == 'Quran')
        tafseer_count = sum(1 for meta in results['metadatas'][0] if meta.get('category') == 'Tafseer')
        
        # Get available tafseers for this surah
        available_tafseers = set()
        for meta in results['metadatas'][0]:
            if meta.get('category') == 'Tafseer' and meta.get('tafseer_name'):
                available_tafseers.add(meta['tafseer_name'])
        
        surah_info.update({
            "verses_available": verse_count,
            "tafseer_entries": tafseer_count,
            "available_tafseers": list(available_tafseers),
            "status": f"{'Complete' if verse_count > 0 else 'No'} Quranic text, {len(available_tafseers)} tafseer(s) available"
        })
        
        return surah_info
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)