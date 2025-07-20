# quran_loader.py - Load complete Quran into ChromaDB
# This will download and process the entire Quran (6,236 verses)

import requests
import json
import chromadb
from chromadb.utils import embedding_functions
import time
from typing import Dict, List

def download_quran_data():
    """Download complete Quran from API"""
    print("Downloading complete Quran data...")
    
    # Using Al-Quran API - free and reliable
    base_url = "https://api.alquran.cloud/v1"
    
    try:
        # Get Quran in English (Sahih International translation)
        response = requests.get(f"{base_url}/quran/en.sahih")
        
        if response.status_code == 200:
            data = response.json()
            if data['code'] == 200:
                print("✅ Quran data downloaded successfully!")
                return data['data']
        
        print("❌ Failed to download from API")
        return None
        
    except Exception as e:
        print(f"Error downloading Quran: {e}")
        return None

def process_quran_for_embedding(quran_data) -> List[Dict]:
    """Process Quran data for ChromaDB embedding"""
    processed_verses = []
    
    print("Processing Quran verses for embedding...")
    
    for surah in quran_data['surahs']:
        surah_number = surah['number']
        surah_name = surah['name']
        surah_english_name = surah['englishName']
        surah_revelation_type = surah['revelationType']
        
        for ayah in surah['ayahs']:
            verse_data = {
                'id': f"quran_{surah_number}_{ayah['numberInSurah']}",
                'text': ayah['text'],
                'surah_number': surah_number,
                'verse_number': ayah['numberInSurah'],
                'surah_name': surah_name,
                'surah_english_name': surah_english_name,
                'revelation_type': surah_revelation_type,
                'source': f"Quran {surah_number}:{ayah['numberInSurah']}",
                'category': 'Quran',
                'full_reference': f"Surah {surah_english_name} ({surah_number}:{ayah['numberInSurah']})"
            }
            processed_verses.append(verse_data)
    
    print(f"✅ Processed {len(processed_verses)} verses!")
    return processed_verses

def load_quran_to_chromadb(processed_verses: List[Dict], collection_name: str = "islamic_texts"):
    """Load processed Quran verses into ChromaDB"""
    
    # Initialize Chroma client
    chroma_client = chromadb.PersistentClient(path="./islamic_knowledge_db")
    
    # Use sentence transformers for embeddings
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Get or create collection
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef
    )
    
    print(f"Starting to load {len(processed_verses)} verses into ChromaDB...")
    
    # Prepare data for batch insertion
    texts = []
    metadatas = []
    ids = []
    
    for verse in processed_verses:
        texts.append(verse['text'])
        ids.append(verse['id'])
        
        # Metadata for each verse
        metadata = {
            'source': verse['source'],
            'category': verse['category'],
            'surah_number': str(verse['surah_number']),
            'verse_number': str(verse['verse_number']),
            'surah_name': verse['surah_name'],
            'surah_english_name': verse['surah_english_name'],
            'revelation_type': verse['revelation_type'],
            'full_reference': verse['full_reference']
        }
        metadatas.append(metadata)
    
    # Batch insert (ChromaDB can handle this efficiently)
    batch_size = 100
    total_batches = len(texts) // batch_size + (1 if len(texts) % batch_size else 0)
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        
        try:
            collection.add(
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            
            current_batch = (i // batch_size) + 1
            print(f"✅ Loaded batch {current_batch}/{total_batches} ({len(batch_texts)} verses)")
            
        except Exception as e:
            print(f"❌ Error loading batch {current_batch}: {e}")
            continue
    
    print(f"🎉 Alhamdulillah! Successfully loaded complete Quran!")
    print(f"Total verses in database: {collection.count()}")
    
    return collection

def test_quran_search(collection, test_queries: List[str]):
    """Test the loaded Quran with some search queries"""
    print("\n" + "="*50)
    print("Testing Quran search capabilities...")
    print("="*50)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 30)
        
        try:
            results = collection.query(
                query_texts=[query],
                n_results=3
            )
            
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                relevance = (1 - distance) * 100
                
                print(f"Result {i+1} (Relevance: {relevance:.1f}%)")
                print(f"Source: {metadata['full_reference']}")
                print(f"Text: {doc[:100]}..." if len(doc) > 100 else f"Text: {doc}")
                print()
                
        except Exception as e:
            print(f"Error searching for '{query}': {e}")

def main():
    """Main function to download and load complete Quran"""
    print("🕌 Loading Complete Quran into Islamic Chatbot")
    print("=" * 50)
    
    # Step 1: Download Quran data
    quran_data = download_quran_data()
    if not quran_data:
        print("❌ Failed to download Quran data. Exiting...")
        return
    
    # Step 2: Process for embedding
    processed_verses = process_quran_for_embedding(quran_data)
    
    # Step 3: Load into ChromaDB
    collection = load_quran_to_chromadb(processed_verses)
    
    # Step 4: Test with sample queries
    test_queries = [
        "patience and perseverance",
        "prayer and worship", 
        "charity and helping poor",
        "forgiveness and mercy",
        "guidance and wisdom",
        "paradise and heaven",
        "fear Allah and piety"
    ]
    
    test_quran_search(collection, test_queries)
    
    print("\n🎉 Complete Quran loading finished!")
    print("Your Islamic chatbot now has access to all 6,236+ verses!")
    print("Run your main.py server and test with more complex questions.")

if __name__ == "__main__":
    main()

# Alternative: Load from local JSON file
def load_from_local_json(json_file_path: str):
    """Load Quran from a local JSON file if you have one"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading local JSON: {e}")
        return None

# Usage:
# python quran_loader.py