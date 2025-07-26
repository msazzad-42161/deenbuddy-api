# tafseer_loader.py - Load complete Quran Tafseer into ChromaDB
# This will download and process multiple tafseers including Ibn Kathir, Jalalayn, etc.

import requests
import json
import chromadb
from chromadb.utils import embedding_functions
import time
from typing import Dict, List, Optional
import re

def get_available_tafseers():
    """Get list of available tafseers from API"""
    print("Fetching available tafseers...")
    
    try:
        # Al-Quran API has multiple tafseers
        response = requests.get("https://api.alquran.cloud/v1/edition?format=text&type=tafsir")
        
        if response.status_code == 200:
            data = response.json()
            if data['code'] == 200:
                tafseers = data['data']
                print(f"✅ Found {len(tafseers)} available tafseers!")
                
                # Display available tafseers
                print("\nAvailable Tafseers:")
                for tafseer in tafseers[:10]:  # Show first 10
                    print(f"- {tafseer['name']} ({tafseer['identifier']})")
                    print(f"  Language: {tafseer['language']}, Type: {tafseer['type']}")
                
                return tafseers
        
        return []
        
    except Exception as e:
        print(f"Error fetching tafseers: {e}")
        return []

def download_tafseer_data(tafseer_identifier: str, tafseer_name: str):
    """Download specific tafseer from API"""
    print(f"Downloading {tafseer_name}...")
    
    try:
        # Get complete tafseer
        response = requests.get(f"https://api.alquran.cloud/v1/quran/{tafseer_identifier}")
        
        if response.status_code == 200:
            data = response.json()
            if data['code'] == 200:
                print(f"✅ {tafseer_name} downloaded successfully!")
                return data['data'], tafseer_name
        
        print(f"❌ Failed to download {tafseer_name}")
        return None, None
        
    except Exception as e:
        print(f"Error downloading {tafseer_name}: {e}")
        return None, None

def download_multiple_tafseers():
    """Download multiple popular tafseers"""
    
    # Popular English and Arabic tafseers
    popular_tafseers = [
        ("en.jalalayn", "Tafsir al-Jalalayn (English)"),
        ("ar.jalalayn", "Tafsir al-Jalalayn (Arabic)"),
        ("en.ahmedali", "Ahmed Ali Commentary"),
        ("en.ahmedrazakhan", "Ahmed Raza Khan Commentary"),
        # Add more as needed
    ]
    
    all_tafseer_data = []
    
    for identifier, name in popular_tafseers:
        try:
            tafseer_data, tafseer_name = download_tafseer_data(identifier, name)
            if tafseer_data:
                all_tafseer_data.append({
                    'identifier': identifier,
                    'name': name,
                    'data': tafseer_data
                })
            time.sleep(1)  # Be respectful to API
        except Exception as e:
            print(f"Skipping {name}: {e}")
            continue
    
    return all_tafseer_data

def process_tafseer_for_embedding(tafseer_collection: List[Dict]) -> List[Dict]:
    """Process tafseer data for ChromaDB embedding"""
    processed_entries = []
    
    print("Processing tafseer data for embedding...")
    
    for tafseer_info in tafseer_collection:
        tafseer_name = tafseer_info['name']
        tafseer_id = tafseer_info['identifier']
        tafseer_data = tafseer_info['data']
        
        print(f"Processing {tafseer_name}...")
        
        for surah in tafseer_data['surahs']:
            surah_number = surah['number']
            surah_name = surah['name']
            surah_english_name = surah['englishName']
            surah_revelation_type = surah['revelationType']
            
            for ayah in surah['ayahs']:
                # Skip if no tafseer text
                if not ayah['text'] or ayah['text'].strip() == "":
                    continue
                
                # Clean up tafseer text
                tafseer_text = clean_tafseer_text(ayah['text'])
                
                if len(tafseer_text) < 10:  # Skip very short entries
                    continue
                
                tafseer_entry = {
                    'id': f"tafseer_{tafseer_id}_{surah_number}_{ayah['numberInSurah']}",
                    'text': tafseer_text,
                    'surah_number': surah_number,
                    'verse_number': ayah['numberInSurah'],
                    'surah_name': surah_name,
                    'surah_english_name': surah_english_name,
                    'revelation_type': surah_revelation_type,
                    'tafseer_name': tafseer_name,
                    'tafseer_identifier': tafseer_id,
                    'source': f"Tafseer {surah_number}:{ayah['numberInSurah']} - {tafseer_name}",
                    'category': 'Tafseer',
                    'full_reference': f"Tafseer of Surah {surah_english_name} ({surah_number}:{ayah['numberInSurah']}) - {tafseer_name}"
                }
                processed_entries.append(tafseer_entry)
        
        print(f"✅ Processed {tafseer_name}")
    
    print(f"✅ Total tafseer entries processed: {len(processed_entries)}")
    return processed_entries

def clean_tafseer_text(text: str) -> str:
    """Clean and format tafseer text"""
    if not text:
        return ""
    
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common prefixes/suffixes that don't add value
    text = text.strip()
    
    # Remove empty parentheses
    text = re.sub(r'\(\s*\)', '', text)
    
    return text

def load_tafseer_to_chromadb(processed_entries: List[Dict], collection_name: str = "islamic_texts"):
    """Load processed tafseer entries into ChromaDB alongside Quran verses"""
    
    # Initialize Chroma client
    chroma_client = chromadb.PersistentClient(path="./islamic_knowledge_db")
    
    # Use sentence transformers for embeddings
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Get existing collection (should already have Quran verses)
    try:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=sentence_transformer_ef
        )
        print(f"✅ Found existing collection with {collection.count()} entries")
    except Exception:
        # Create new collection if doesn't exist
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=sentence_transformer_ef
        )
        print("✅ Created new collection for Islamic texts")
    
    print(f"Starting to load {len(processed_entries)} tafseer entries into ChromaDB...")
    
    # Prepare data for batch insertion
    texts = []
    metadatas = []
    ids = []
    
    for entry in processed_entries:
        texts.append(entry['text'])
        ids.append(entry['id'])
        
        # Metadata for each tafseer entry
        metadata = {
            'source': entry['source'],
            'category': entry['category'],
            'surah_number': str(entry['surah_number']),
            'verse_number': str(entry['verse_number']),
            'surah_name': entry['surah_name'],
            'surah_english_name': entry['surah_english_name'],
            'revelation_type': entry['revelation_type'],
            'full_reference': entry['full_reference'],
            'tafseer_name': entry['tafseer_name'],
            'tafseer_identifier': entry['tafseer_identifier']
        }
        metadatas.append(metadata)
    
    # Batch insert
    batch_size = 50  # Smaller batches for tafseer (often longer text)
    total_batches = len(texts) // batch_size + (1 if len(texts) % batch_size else 0)
    
    successful_entries = 0
    
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
            
            successful_entries += len(batch_texts)
            current_batch = (i // batch_size) + 1
            print(f"✅ Loaded batch {current_batch}/{total_batches} ({len(batch_texts)} entries)")
            
        except Exception as e:
            print(f"❌ Error loading batch {current_batch}: {e}")
            # Try individual entries in case of conflicts
            for j, (text, metadata, id_val) in enumerate(zip(batch_texts, batch_metadatas, batch_ids)):
                try:
                    collection.add(
                        documents=[text],
                        metadatas=[metadata],
                        ids=[id_val]
                    )
                    successful_entries += 1
                except Exception as individual_error:
                    print(f"Skipping entry {id_val}: {individual_error}")
                    continue
    
    final_count = collection.count()
    print(f"🎉 Alhamdulillah! Successfully loaded tafseer data!")
    print(f"Successfully added: {successful_entries} tafseer entries")
    print(f"Total entries in database: {final_count}")
    
    return collection

def test_tafseer_search(collection, test_queries: List[str]):
    """Test the loaded tafseer with search queries"""
    print("\n" + "="*60)
    print("Testing Combined Quran + Tafseer Search")
    print("="*60)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        try:
            # Search with higher result count to get both verse and tafseer
            results = collection.query(
                query_texts=[query],
                n_results=6
            )
            
            verses_found = 0
            tafseer_found = 0
            
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                relevance = (1 - distance) * 100
                
                category = metadata.get('category', 'Unknown')
                
                if category == 'Quran':
                    verses_found += 1
                elif category == 'Tafseer':
                    tafseer_found += 1
                
                print(f"{category} Result {i+1} (Relevance: {relevance:.1f}%)")
                print(f"Source: {metadata['full_reference']}")
                
                if category == 'Tafseer':
                    print(f"Tafseer: {metadata.get('tafseer_name', 'Unknown')}")
                
                preview_text = doc[:150] + "..." if len(doc) > 150 else doc
                print(f"Text: {preview_text}")
                print()
            
            print(f"Summary: Found {verses_found} verses and {tafseer_found} tafseer entries")
            print("-" * 40)
                
        except Exception as e:
            print(f"Error searching for '{query}': {e}")

def main():
    """Main function to download and load tafseer"""
    print("🕌 Loading Quran Tafseer into Islamic Chatbot")
    print("=" * 60)
    
    # Option 1: Download multiple popular tafseers
    print("Downloading popular tafseers...")
    tafseer_collection = download_multiple_tafseers()
    
    if not tafseer_collection:
        print("❌ No tafseer data downloaded. Exiting...")
        return
    
    print(f"\n✅ Downloaded {len(tafseer_collection)} tafseers")
    
    # Process for embedding
    processed_entries = process_tafseer_for_embedding(tafseer_collection)
    
    if not processed_entries:
        print("❌ No tafseer entries processed. Exiting...")
        return
    
    # Load into ChromaDB (same collection as Quran verses)
    collection = load_tafseer_to_chromadb(processed_entries)
    
    # Test with sample queries
    test_queries = [
        "patience and hardship",
        "prayer importance", 
        "charity and wealth",
        "forgiveness from Allah",
        "guidance for believers",
        "paradise description",
        "fear of Allah taqwa"
    ]
    
    test_tafseer_search(collection, test_queries)
    
    print("\n🎉 Tafseer loading completed!")
    print("Your Islamic chatbot now has:")
    print("- Complete Quran verses (6,200+ verses)")
    print("- Multiple tafseer commentaries")
    print("- Enhanced scholarly context for better answers")
    print("\nRun your main.py server and ask detailed Islamic questions!")

def load_custom_tafseer(tafseer_identifier: str, tafseer_name: str):
    """Load a specific tafseer by identifier"""
    print(f"Loading specific tafseer: {tafseer_name}")
    
    tafseer_data, name = download_tafseer_data(tafseer_identifier, tafseer_name)
    if not tafseer_data:
        print(f"Failed to download {tafseer_name}")
        return
    
    tafseer_collection = [{
        'identifier': tafseer_identifier,
        'name': name,
        'data': tafseer_data
    }]
    
    processed_entries = process_tafseer_for_embedding(tafseer_collection)
    collection = load_tafseer_to_chromadb(processed_entries)
    
    print(f"✅ Successfully loaded {tafseer_name}!")
    return collection

if __name__ == "__main__":
    # Option to load all popular tafseers
    main()
    
    # Or load specific tafseer:
    # load_custom_tafseer("en.jalalayn", "Tafsir al-Jalalayn (English)")