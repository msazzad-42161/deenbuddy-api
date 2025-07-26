# hadith_loader.py - Load Sahih Bukhari Hadiths (and explanations) into ChromaDB
import json
import chromadb
from chromadb.utils import embedding_functions

def load_bukhari_hadiths_to_chromadb(hadith_json_path: str, explanation_json_path: str = None, collection_name: str = "islamic_texts"):
    """Load Sahih Bukhari hadiths (and optional explanations) into ChromaDB."""
    print(f"Loading Sahih Bukhari hadiths from {hadith_json_path} ...")
    try:
        with open(hadith_json_path, 'r', encoding='utf-8') as f:
            hadith_data = json.load(f)
    except Exception as e:
        print(f"Error loading Bukhari JSON: {e}")
        return None

    explanations = None
    if explanation_json_path:
        try:
            with open(explanation_json_path, 'r', encoding='utf-8') as f:
                explanations = json.load(f)
        except Exception as e:
            print(f"Error loading explanations JSON: {e}")
            explanations = None

    # Prepare entries for ChromaDB
    entries = []
    hadiths = hadith_data.get('hadiths', hadith_data.get('data', []))
    for h in hadiths:
        hadith_number = h.get('number') or h.get('hadithnumber') or h.get('id')
        book = h.get('book', h.get('bookNumber', ''))
        chapter = h.get('chapter', h.get('chapterNumber', ''))
        text = h.get('text', h.get('body', ''))
        narrator = h.get('narrator', h.get('by', ''))
        reference = h.get('reference', '')
        # Try to get explanation if available
        explanation_text = ''
        if explanations:
            # Try to match by hadith number
            exp = next((e for e in explanations if str(e.get('number', e.get('hadithnumber', e.get('id', ''))) ) == str(hadith_number)), None)
            if exp:
                explanation_text = exp.get('explanation', exp.get('sharh', ''))

        entry = {
            'id': f"bukhari_{hadith_number}",
            'text': text,
            'book': str(book),
            'chapter': str(chapter),
            'hadith_number': str(hadith_number),
            'narrator': narrator,
            'reference': reference,
            'category': 'Hadith',
            'source': f"Sahih Bukhari {book}:{hadith_number}",
            'full_reference': f"Sahih Bukhari (Book {book}, Hadith {hadith_number})",
            'explanation': explanation_text
        }
        entries.append(entry)

    print(f"Processing {len(entries)} hadith entries for embedding ...")

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
    def stringify_if_dict(val):
        if isinstance(val, dict):
            # Convert dict to a readable string, e.g. 'book: 97, hadith: 53'
            return ', '.join(f"{k}: {v}" for k, v in val.items())
        return val

    for entry in entries:
        texts.append(entry['text'])
        ids.append(entry['id'])
        metadata = {
            'source': entry['source'],
            'category': entry['category'],
            'book': entry['book'],
            'chapter': entry['chapter'],
            'hadith_number': entry['hadith_number'],
            'narrator': entry['narrator'],
            'reference': stringify_if_dict(entry['reference']),
            'full_reference': entry['full_reference'],
        }
        if entry['explanation']:
            metadata['explanation'] = entry['explanation']
        metadatas.append(metadata)

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
    print(f"🎉 Successfully loaded {successful} Sahih Bukhari hadiths into ChromaDB!")
    print(f"Total entries in collection: {collection.count()}")
    return collection


if __name__ == "__main__":
    # Option to load all popular tafseers
    load_bukhari_hadiths_to_chromadb(hadith_json_path="./rag_data/eng-bukhari.json")