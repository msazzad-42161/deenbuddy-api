# db_diagnostic.py - Diagnose your Islamic knowledge database
import chromadb
from chromadb.utils import embedding_functions
import json
from collections import Counter

def diagnose_database():
    """Comprehensive database diagnosis"""
    print("🔍 DIAGNOSING ISLAMIC KNOWLEDGE DATABASE")
    print("=" * 60)
    
    try:
        # Initialize Chroma client
        chroma_client = chromadb.PersistentClient(path="./islamic_knowledge_db")
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        collection = chroma_client.get_or_create_collection(
            name="islamic_texts",
            embedding_function=sentence_transformer_ef
        )
        
        total_count = collection.count()
        print(f"📊 Total entries in database: {total_count}")
        
        if total_count == 0:
            print("❌ DATABASE IS EMPTY!")
            print("Solution: Run quran_loader.py, tafseer_loader.py, and tirmidhi_loader.py to populate the database")
            return
        
        # Get a large sample to analyze
        sample_size = min(2000, total_count)  # Increased sample size for better analysis
        print(f"🔬 Analyzing sample of {sample_size} entries...")
        
        results = collection.query(
            query_texts=["Allah guidance prayer"],
            n_results=sample_size
        )
        
        # Analyze categories
        categories = Counter()
        surah_distribution = Counter()
        tafseer_distribution = Counter()
        hadith_distribution = Counter()
        hadith_collections = Counter()
        
        print("\n📋 DETAILED ANALYSIS:")
        print("-" * 40)
        
        for i, metadata in enumerate(results['metadatas'][0]):
            category = metadata.get('category', 'Unknown')
            categories[category] += 1
            
            if category == 'Quran':
                surah_name = metadata.get('surah_english_name', 'Unknown')
                surah_distribution[surah_name] += 1
            elif category == 'Tafseer':
                tafseer_name = metadata.get('tafseer_name', 'Unknown')
                tafseer_distribution[tafseer_name] += 1
            elif category == 'Hadith':
                # Handle new hadith format
                collection_name = metadata.get('collection', 
                                            metadata.get('source', 'Unknown'))
                hadith_collections[collection_name] += 1
                
                # Also track by source for backward compatibility
                source = metadata.get('source', 'Unknown Hadith')
                hadith_distribution[source] += 1
        
        # Display results
        print("📊 CATEGORY BREAKDOWN:")
        for category, count in categories.most_common():
            percentage = (count / sample_size) * 100
            print(f"   {category}: {count} entries ({percentage:.1f}%)")
        
        if categories['Quran'] > 0:
            print(f"\n📖 QURAN SURAHS (Top 10):")
            for surah, count in surah_distribution.most_common(10):
                print(f"   {surah}: {count} verses")
        
        if categories['Tafseer'] > 0:
            print(f"\n💭 TAFSEER SOURCES:")
            for tafseer, count in tafseer_distribution.most_common():
                print(f"   {tafseer}: {count} entries")
        
        if categories['Hadith'] > 0:
            print(f"\n📜 HADITH COLLECTIONS:")
            for hadith_col, count in hadith_collections.most_common():
                print(f"   {hadith_col}: {count} entries")
            
            # Show detailed hadith sources if available
            print(f"\n📚 HADITH SOURCES (Detailed):")
            for source, count in sorted(hadith_distribution.items()):
                if 'Hadith' in source or any(x in source.lower() for x in ['bukhari', 'tirmidhi', 'muslim', 'abu', 'nasai', 'majah']):
                    print(f"   {source}: {count} entries")
        
        return collection, categories
        
    except Exception as e:
        print(f"❌ Error accessing database: {e}")
        return None, None

def test_specific_searches(collection):
    """Test specific search queries"""
    print(f"\n🔍 TESTING SPECIFIC SEARCHES")
    print("=" * 60)
    
    test_cases = [
        {
            "query": "musa moses prayer 40 years",
            "expected": "Should find Quran verses about Moses",
            "category_filter": None
        },
        {
            "query": "sisters rights haqooq",
            "expected": "Should find verses about women's rights",
            "category_filter": None
        },
        {
            "query": "prayer salah",
            "expected": "Should find verses and hadiths about prayer",
            "category_filter": "Quran"
        },
        {
            "query": "charity zakat",
            "expected": "Should find verses about charity",
            "category_filter": "Quran"
        },
        {
            "query": "ablution wudu purification",
            "expected": "Should find hadiths about ablution",
            "category_filter": "Hadith"
        },
        {
            "query": "Prophet Muhammad companions",
            "expected": "Should find hadiths about the Prophet and companions",
            "category_filter": "Hadith"
        }
    ]
    
    for test in test_cases:
        print(f"\n🔎 Query: '{test['query']}'")
        print(f"Expected: {test['expected']}")
        print("-" * 30)
        
        try:
            # Build query parameters
            query_params = {
                "query_texts": [test['query']],
                "n_results": 10
            }
            
            # Add category filter if specified
            if test['category_filter']:
                query_params["where"] = {"category": test['category_filter']}
            
            results = collection.query(**query_params)
            
            categories_found = Counter()
            quran_verses = []
            tafseer_entries = []
            hadith_entries = []
            
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                relevance = (1 - distance) * 100
                
                category = metadata.get('category', 'Unknown')
                categories_found[category] += 1
                
                if category == 'Quran':
                    quran_verses.append({
                        'reference': metadata.get('full_reference', 'Unknown'),
                        'surah': metadata.get('surah_english_name', ''),
                        'relevance': f"{relevance:.1f}%",
                        'text_preview': doc[:100] + "..." if len(doc) > 100 else doc
                    })
                elif category == 'Tafseer':
                    tafseer_entries.append({
                        'reference': metadata.get('full_reference', 'Unknown'),
                        'tafseer': metadata.get('tafseer_name', ''),
                        'relevance': f"{relevance:.1f}%"
                    })
                elif category == 'Hadith':
                    hadith_entries.append({
                        'reference': metadata.get('full_reference', 'Unknown'),
                        'collection': metadata.get('collection', metadata.get('source', '')),
                        'narrator': metadata.get('narrator', ''),
                        'hadith_id': metadata.get('hadith_id', ''),
                        'relevance': f"{relevance:.1f}%",
                        'text_preview': doc[:100] + "..." if len(doc) > 100 else doc
                    })
            
            print("📊 Results by category:")
            for cat, count in categories_found.items():
                print(f"   {cat}: {count}")
            
            if quran_verses:
                print("📖 Top Quran results:")
                for verse in quran_verses[:3]:
                    print(f"   • {verse['reference']} ({verse['relevance']})")
                    print(f"     {verse['text_preview']}")
            
            if hadith_entries:
                print("📜 Top Hadith results:")
                for hadith in hadith_entries[:3]:
                    narrator_info = f" - {hadith['narrator']}" if hadith['narrator'] else ""
                    print(f"   • {hadith['collection']} #{hadith['hadith_id']}{narrator_info} ({hadith['relevance']})")
                    if hadith['text_preview']:
                        print(f"     {hadith['text_preview']}")
            
            if not quran_verses and test['expected'].startswith("Should find verses"):
                print("⚠️  WARNING: Expected Quran verses but found none!")
            
            if not hadith_entries and test['expected'].startswith("Should find hadiths"):
                print("⚠️  WARNING: Expected hadiths but found none!")
            
        except Exception as e:
            print(f"❌ Search failed: {e}")

def check_hadith_collections(collection):
    """Specifically check hadith collections"""
    print(f"\n📜 HADITH COLLECTIONS VERIFICATION")
    print("=" * 60)
    
    try:
        # Check for specific hadith collections
        expected_collections = ['Tirmidhi', 'Bukhari', 'Muslim', 'Abu Dawud', 'Nasai', 'Ibn Majah']
        
        for collection_name in expected_collections:
            print(f"\n🔍 Checking for {collection_name} hadiths...")
            
            # Try multiple search patterns
            search_queries = [
                {"where": {"collection": collection_name}},
                {"where": {"collection": collection_name.lower()}},
                {"query_texts": [collection_name], "where": {"category": "Hadith"}}
            ]
            
            found = False
            for query in search_queries:
                try:
                    results = collection.query(n_results=5, **query)
                    if results['documents'][0]:
                        print(f"✅ Found {len(results['documents'][0])} {collection_name} hadiths")
                        
                        # Show sample
                        metadata = results['metadatas'][0][0]
                        print(f"   Sample: {metadata.get('full_reference', 'Unknown')}")
                        if metadata.get('narrator'):
                            print(f"   Narrator: {metadata.get('narrator')}")
                        found = True
                        break
                except:
                    continue
            
            if not found:
                print(f"❌ No {collection_name} hadiths found")
        
        # Test hadith-specific searches
        hadith_concepts = [
            "prayer times",
            "ablution washing",
            "charity giving",
            "fasting ramadan",
            "pilgrimage hajj"
        ]
        
        print(f"\n🔍 Testing hadith-specific concepts:")
        for concept in hadith_concepts:
            results = collection.query(
                query_texts=[concept],
                n_results=3,
                where={"category": "Hadith"}
            )
            
            if results['documents'][0]:
                print(f"✅ '{concept}': Found {len(results['documents'][0])} hadiths")
                metadata = results['metadatas'][0][0]
                collection_name = metadata.get('collection', 'Unknown')
                hadith_id = metadata.get('hadith_id', 'Unknown')
                print(f"   Best match: {collection_name} #{hadith_id}")
            else:
                print(f"❌ '{concept}': No hadiths found")
        
    except Exception as e:
        print(f"❌ Hadith verification failed: {e}")

def check_quran_content(collection):
    """Specifically check Quran content"""
    print(f"\n📖 QURAN CONTENT VERIFICATION")
    print("=" * 60)
    
    try:
        # Search for well-known verses
        famous_verses = [
            "In the name of Allah most gracious most merciful",  # Bismillah concept
            "There is no god but Allah",  # Shahada concept
            "Establish prayer",  # Prayer
            "Moses said to his people",  # Moses stories
            "And it is He who created the heavens and earth"  # Creation verses
        ]
        
        for verse_concept in famous_verses:
            print(f"\n🔍 Searching for: '{verse_concept}'")
            results = collection.query(
                query_texts=[verse_concept],
                n_results=5,
                where={"category": "Quran"}  # Filter only Quran
            )
            
            if results['documents'][0]:
                print(f"✅ Found {len(results['documents'][0])} Quran matches")
                for i, doc in enumerate(results['documents'][0][:2]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    relevance = (1 - distance) * 100
                    print(f"   • {metadata.get('full_reference', 'Unknown')} ({relevance:.1f}%)")
                    print(f"     {doc[:80]}...")
            else:
                print("❌ No Quran matches found!")
        
    except Exception as e:
        print(f"❌ Quran verification failed: {e}")

def main():
    """Run complete diagnosis"""
    collection, categories = diagnose_database()
    
    if collection and categories:
        if categories.get('Quran', 0) == 0:
            print("\n🚨 CRITICAL ISSUE: NO QURAN VERSES FOUND!")
            print("SOLUTION: Run 'python quran_loader.py' to load Quran verses")
        else:
            check_quran_content(collection)
        
        if categories.get('Hadith', 0) == 0:
            print("\n🚨 CRITICAL ISSUE: NO HADITHS FOUND!")
            print("SOLUTION: Run 'python tirmidhi_loader.py' or other hadith loaders")
        else:
            check_hadith_collections(collection)
        
        # Always run general search tests
        test_specific_searches(collection)
        
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 40)
        
        if categories.get('Quran', 0) == 0:
            print("1. ❗ URGENT: Load Quran verses with quran_loader.py")
        else:
            print("1. ✅ Quran verses are present")
        
        if categories.get('Hadith', 0) == 0:
            print("2. ❗ URGENT: Load Hadiths with tirmidhi_loader.py or other hadith loaders")
        else:
            print("2. ✅ Hadith collections are present")
            
            if categories.get('Hadith', 0) > categories.get('Quran', 0) * 3:
                print("   ⚠️  You have significantly more Hadiths than Quran verses")
                print("      This may bias search results toward Hadiths")
        
        if categories.get('Tafseer', 0) == 0:
            print("3. 💡 Consider adding Tafseer with tafseer_loader.py for better context")
        else:
            print("3. ✅ Tafseer commentary is present")
        
        print("\n🔧 DEBUGGING TIPS:")
        print("- Use /search/{query}?category=quran to force Quran-only search")
        print("- Use /search/{query}?category=hadith to force Hadith-only search")
        print("- Check if your question terms match the English translation style")
        print("- Try broader terms like 'Moses' instead of 'Musa'")
        print("- Use multiple search terms: 'Moses Pharaoh Egypt'")
        print("- For hadiths, try terms like 'Prophet said' or 'narrated by'")
        
        print(f"\n📈 DATABASE HEALTH SCORE:")
        score = 0
        if categories.get('Quran', 0) > 0: score += 40
        if categories.get('Hadith', 0) > 0: score += 35
        if categories.get('Tafseer', 0) > 0: score += 25
        
        if score >= 90:
            print(f"🏆 EXCELLENT ({score}/100) - All major Islamic text types present!")
        elif score >= 70:
            print(f"✅ GOOD ({score}/100) - Core texts present, consider adding missing types")
        elif score >= 40:
            print(f"⚠️  FAIR ({score}/100) - Missing important text types")
        else:
            print(f"❌ POOR ({score}/100) - Critical texts missing!")

if __name__ == "__main__":
    main()