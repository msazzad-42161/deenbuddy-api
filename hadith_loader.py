# tirmidhi_loader.py - Load Tirmidhi Hadiths into ChromaDB
import json
import chromadb
from chromadb.utils import embedding_functions

def load_tirmidhi_hadiths_to_chromadb(hadith_json_path: str, collection_name: str = "islamic_texts"):
    """Load Tirmidhi hadiths into ChromaDB using the hadith-json format."""
    print(f"Loading Tirmidhi hadiths from {hadith_json_path} ...")
    try:
        with open(hadith_json_path, 'r', encoding='utf-8') as f:
            hadith_data = json.load(f)
    except Exception as e:
        print(f"Error loading Tirmidhi JSON: {e}")
        return None

    # Prepare entries for ChromaDB
    entries = []
    
    # The hadith-json format stores hadiths directly as a list
    hadiths = hadith_data if isinstance(hadith_data, list) else hadith_data.get('hadiths', [])
    
    for h in hadiths:
        hadith_id = h.get('id')
        chapter_id = h.get('chapterId')
        book_id = h.get('bookId')
        arabic_text = h.get('arabic', '')
        english_data = h.get('english', {})
        narrator = english_data.get('narrator', '') if isinstance(english_data, dict) else ''
        english_text = english_data.get('text', '') if isinstance(english_data, dict) else str(english_data)

        # Create combined text for better searchability
        combined_text = f"{english_text}"
        if narrator:
            combined_text = f"Narrated by {narrator}: {english_text}"

        entry = {
            'id': f"tirmidhi_{hadith_id}",
            'text': combined_text,
            'english_text': english_text,
            'arabic_text': arabic_text,
            'narrator': narrator,
            'hadith_id': str(hadith_id),
            'chapter_id': str(chapter_id),
            'book_id': str(book_id),
            'category': 'Hadith',
            'source': f"Jami` at-Tirmidhi {hadith_id}",
            'full_reference': f"Jami` at-Tirmidhi (Hadith {hadith_id})",
            'collection': 'Tirmidhi'
        }
        entries.append(entry)

    print(f"Processing {len(entries)} Tirmidhi hadith entries for embedding ...")

    # Initialize Chroma client
    chroma_client = chromadb.PersistentClient(path="./islamic_knowledge_db")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef
    )

    # Prepare for batch insert
    texts = []
    metadatas = []
    ids = []

    for entry in entries:
        texts.append(entry['text'])
        ids.append(entry['id'])
        
        metadata = {
            'source': entry['source'],
            'category': entry['category'],
            'collection': entry['collection'],
            'hadith_id': entry['hadith_id'],
            'chapter_id': entry['chapter_id'],
            'book_id': entry['book_id'],
            'narrator': entry['narrator'],
            'full_reference': entry['full_reference'],
            'english_text': entry['english_text'],
            'arabic_text': entry['arabic_text']
        }
        metadatas.append(metadata)

    # Batch insert with error handling
    batch_size = 50
    total_batches = len(texts) // batch_size + (1 if len(texts) % batch_size else 0)
    successful = 0
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        try:
            collection.add(
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            successful += len(batch_texts)
            print(f"✅ Loaded batch {(i//batch_size)+1}/{total_batches} ({len(batch_texts)} hadiths)")
        except Exception as e:
            print(f"❌ Error loading batch {(i//batch_size)+1}: {e}")
            continue
    
    print(f"🎉 Successfully loaded {successful} Tirmidhi hadiths into ChromaDB!")
    print(f"Total entries in collection: {collection.count()}")
    return collection


def load_multiple_hadith_collections(collections_config: dict, db_collection_name: str = "islamic_texts"):
    """
    Load multiple hadith collections into ChromaDB.
    
    Args:
        collections_config: Dict with format {"collection_name": "path_to_json"}
        db_collection_name: Name of the ChromaDB collection
        
    Example:
        config = {
            "tirmidhi": "./rag_data/tirmidhi.json",
            "bukhari": "./rag_data/bukhari.json",
            "muslim": "./rag_data/muslim.json"
        }
    """
    chroma_client = chromadb.PersistentClient(path="./islamic_knowledge_db")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = chroma_client.get_or_create_collection(
        name=db_collection_name,
        embedding_function=sentence_transformer_ef
    )
    
    for collection_name, json_path in collections_config.items():
        print(f"\n📚 Loading {collection_name.title()} collection...")
        load_hadith_collection_generic(json_path, collection_name, collection)
    
    print(f"\n🎉 All collections loaded! Total entries: {collection.count()}")
    return collection


def load_hadith_collection_generic(json_path: str, collection_name: str, chroma_collection):
    """Generic loader for hadith-json format collections."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            hadith_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading {collection_name} JSON: {e}")
        return

    hadiths = hadith_data if isinstance(hadith_data, list) else hadith_data.get('hadiths', [])
    
    texts = []
    metadatas = []
    ids = []
    
    for h in hadiths:
        hadith_id = h.get('id')
        chapter_id = h.get('chapterId')
        book_id = h.get('bookId')
        arabic_text = h.get('arabic', '')
        english_data = h.get('english', {})
        narrator = english_data.get('narrator', '') if isinstance(english_data, dict) else ''
        english_text = english_data.get('text', '') if isinstance(english_data, dict) else str(english_data)

        # Create combined text for better searchability
        combined_text = f"{english_text}"
        if narrator:
            combined_text = f"Narrated by {narrator}: {english_text}"

        texts.append(combined_text)
        ids.append(f"{collection_name}_{hadith_id}")
        
        metadata = {
            'source': f"{collection_name.title()} {hadith_id}",
            'category': 'Hadith',
            'collection': collection_name.title(),
            'hadith_id': str(hadith_id),
            'chapter_id': str(chapter_id),
            'book_id': str(book_id),
            'narrator': narrator,
            'full_reference': f"{collection_name.title()} (Hadith {hadith_id})",
            'english_text': english_text,
            'arabic_text': arabic_text
        }
        metadatas.append(metadata)

    # Batch insert
    batch_size = 50
    successful = 0
    total_batches = len(texts) // batch_size + (1 if len(texts) % batch_size else 0)
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        try:
            chroma_collection.add(
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            successful += len(batch_texts)
            print(f"✅ {collection_name.title()}: Loaded batch {(i//batch_size)+1}/{total_batches}")
        except Exception as e:
            print(f"❌ {collection_name.title()}: Error in batch {(i//batch_size)+1}: {e}")
            continue
    
    print(f"✅ {collection_name.title()}: Successfully loaded {successful} hadiths")


if __name__ == "__main__":
    # Option 1: Load only Tirmidhi
    # load_tirmidhi_hadiths_to_chromadb(hadith_json_path="./rag_data/tirmidhi.json")
    
    # Option 2: Load multiple collections at once
    collections = {
        "tirmidhi": "./rag_data/tirmidhi.json",
        "bukhari": "./rag_data/bukhari.json",
        "muslim": "./rag_data/muslim.json",
        "abudawud": "./rag_data/abudawud.json",
        "nasai": "./rag_data/nasai.json",
        "ibnmajah": "./rag_data/ibnmajah.json"
    }
    load_multiple_hadith_collections(collections)