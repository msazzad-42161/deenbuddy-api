# main.py - Comprehensive Islamic Knowledge API with GitHub Models

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
from typing import List, Dict
import re
from datetime import datetime

# --- GitHub Models Imports ---
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Load environment variables
load_dotenv()

app = FastAPI(title="Islamic Knowledge API - Complete Quran + Tafseer Edition (GitHub Models)")

# --- GitHub Models Setup ---
github_client = None
GITHUB_ENDPOINT = "https://models.github.ai/inference"
GITHUB_MODEL_NAME = "gpt-4o"  # You can also use "gpt-4o-mini" for faster responses

try:
    # Get GitHub token from environment variable
    github_token = os.getenv("GITHUB_TOKEN")
    
    if not github_token:
        raise ValueError("GITHUB_TOKEN environment variable not set. Please get your GitHub token and add it to your .env file.")
    
    # Initialize GitHub Models client
    github_client = ChatCompletionsClient(
        endpoint=GITHUB_ENDPOINT,
        credential=AzureKeyCredential(github_token),
    )
    
    # Test the connection with a simple API call
    try:
        test_response = github_client.complete(
            messages=[
                SystemMessage("You are a helpful assistant."),
                UserMessage("Hello")
            ],
            temperature=0.1,
            top_p=1.0,
            max_tokens=5,
            model=GITHUB_MODEL_NAME
        )
        print(f"✅ Alhamdulillah! GitHub Models ({GITHUB_MODEL_NAME}) configured and tested successfully.")
    except Exception as test_error:
        print(f"⚠️  GitHub Models client created but test call failed: {test_error}")
        print("This might be due to token issues or rate limits. The client will still be available for use.")
    
except Exception as e:
    print(f"❌ ERROR: Failed to configure or initialize GitHub Models. Error details: {e}")
    print("Common fixes:")
    print("1. Make sure you have the Azure AI Inference library: pip install azure-ai-inference")
    print("2. Get your GitHub token from: https://github.com/settings/tokens")
    print("3. Ensure your .env file contains: GITHUB_TOKEN=your_actual_token_here")
    print("4. Check that your token has the necessary permissions")
    github_client = None

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
    greeting: str = "Hey there, buddy! 👋"
    total_texts_searched: int
    verses_found: int = 0
    tafseer_found: int = 0

def search_islamic_knowledge(question: str, n_results: int = 6, include_tafseer: bool = True):
    """Enhanced search for Quran verses and tafseer"""
    try:
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
    """Generate comprehensive response using both Quran and Tafseer by leveraging GitHub Models."""
    
    # Check if GitHub Models client is initialized.
    if github_client is None:
        return (f"Hey there! 😅 I really wanted to help you out with that question, but I'm having some technical "
                f"issues connecting to my AI buddy right now. The GitHub Models setup needs some tweaking - "
                f"probably just need to double-check that GITHUB_TOKEN in the settings. Give me a moment to get "
                f"that sorted, and then I'll be right back to help you out! 🔧")

    if not relevant_texts:
        # Custom message if no relevant texts are found by Chroma
        return "Hey buddy! 😊 So I looked everywhere for you, but honestly couldn't find anything super specific about that exact question in my current collection of Quran verses and Tafseer. Sometimes the way we phrase things just doesn't match up perfectly with what's in the database, you know? I'd totally recommend chatting with your local imam or a trusted Islamic teacher - they'll probably have some great insights for you. Hope that helps, and don't hesitate to try rephrasing your question if you want to give it another shot! 🤝"

    # --- Construct the Prompt for GitHub Models ---
    system_prompt = (
        "You are 'DeenBuddy' - a knowledgeable, fun, and super friendly Muslim friend who's like that childhood buddy "
        "who always had the best answers and made everything sound so simple and relatable. Think of yourself as that "
        "cool, wise friend from the neighborhood who knows Islam really well but explains things in the most chill, "
        "down-to-earth way possible. "
        "\n\nYour personality:\n"
        "- Talk like you're texting your best friend - casual, warm, and genuine\n"
        "- Use modern, friendly language mixed with appropriate Islamic phrases naturally (not forced)\n"
        "- Be encouraging and supportive, like a buddy who's got your back\n"
        "- Make complex Islamic concepts feel easy and relatable\n"
        "- Use examples from everyday life that people can connect with\n"
        "- Sometimes use casual expressions like 'bro', 'my friend', 'you know what I mean?', 'honestly', 'listen'\n"
        "- Be authentic - if something is serious, be serious, but keep it approachable\n"
        "- Always cite your sources properly but in a natural way: [Quran, SurahName:Verse] or [Tafseer Name, Reference]\n"
        "\nRemember: You're not a formal scholar giving a lecture - you're that trusted friend who happens to know "
        "the Deen really well and always gives the best, most heartfelt advice. Keep it real, keep it friendly, "
        "and make your friend feel heard and supported. End with a genuine, warm closing that feels like something "
        "a real friend would say."
    )
    
    user_query_context = []
    user_query_context.append(f"Hey DeenBuddy! Your friend just asked you: \"{question}\"\n")
    user_query_context.append("You've got some solid Islamic knowledge to help them out. Here's what you found "
                              "that's relevant to their question. Use this info to give them a genuine, friendly response "
                              "that feels like you're just chatting over coffee:\n\n")

    sources_for_llm = []
    # Add relevant texts to the prompt
    for i, text_info in enumerate(relevant_texts):
        category = text_info.get("category", "Unknown")
        full_reference = text_info.get("full_reference", "Unknown Reference")
        surah_name = text_info.get("surah_english_name", "")
        text_content = text_info["text"]
        
        # Prepare content for LLM in a more casual way
        if category == "Quran":
            sources_for_llm.append(f"📖 **From the Quran ({surah_name} {full_reference}):** \"{text_content}\"\n")
        elif category == "Tafseer":
            tafseer_name = text_info.get("tafseer_name", "Unknown Tafseer")
            sources_for_llm.append(f"💭 **Scholar's explanation ({tafseer_name} on {full_reference}):** \"{text_content}\"\n")
    
    user_query_context.extend(sources_for_llm)
    
    # Dynamic friendly opening based on current time
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        user_query_context.insert(0, "Morning, buddy! ☀️ Hope you're having a good start to your day.")
    elif 12 <= current_hour < 17:
        user_query_context.insert(0, "Hey there! 🌤️ Hope your afternoon is going well, my friend.")
    else:
        user_query_context.insert(0, "Hey buddy! 🌙 Hope you're doing well this evening.")

    messages = [
        SystemMessage(system_prompt),
        UserMessage("\n".join(user_query_context))
    ]

    try:
        # Call GitHub Models API
        response = github_client.complete(
            messages=messages,
            temperature=0.7,  # Adjust creativity
            top_p=1.0,
            max_tokens=1500,  # GitHub Models can handle longer responses
            model=GITHUB_MODEL_NAME
        )
        
        answer_text = response.choices[0].message.content
        return answer_text

    except Exception as e:
        print(f"Error generating response with GitHub Models: {e}")
        error_str = str(e).lower()
        
        if "rate limit" in error_str or "quota" in error_str:
            return (f"Whoa, hold up buddy! 😅 Looks like I hit some kind of rate limit with the GitHub API. "
                    f"Basically, I got a little too excited helping people out today! Give it a minute and try again, "
                    f"or maybe the GitHub account needs a limits check. I'll be back to normal soon, promise! 🚀")
        elif "token" in error_str or "authentication" in error_str or "unauthorized" in error_str:
            return (f"Uh oh, looks like there's some authentication hiccup with the GitHub connection. 🔐 "
                    f"Probably just need to refresh that GITHUB_TOKEN - you know how these API things can be sometimes! "
                    f"Once that's sorted, I'll be right back to helping you out with your questions! 💪")
        elif "timeout" in error_str:
            return (f"Dang, that took way too long to process! ⏱️ Either your question was super complex (which is awesome!) "
                    f"or the internet's being a bit slow today. Want to try again, maybe with a shorter version of your question? "
                    f"I promise I'll try to be quicker this time! 😊")
        else:
            return (f"Hey buddy, I ran into some technical weirdness while trying to answer your question: \"{question}\". 🤔 "
                    f"Nothing serious, probably just one of those random tech glitches. Give it another shot in a sec? "
                    f"I'm usually way better at this, I promise! Here's what happened: {e}")

@app.on_event("startup")
async def startup_event():
    """Check database and LLM status on startup"""
    count = collection.count()
    
    print("\n--- Startup Checks ---")
    if count > 0:
        try:
            sample = collection.query(query_texts=["Allah"], n_results=min(10, count))
            verse_count = sum(1 for meta in sample['metadatas'][0] if meta.get('category') == 'Quran')
            tafseer_count = sum(1 for meta in sample['metadatas'][0] if meta.get('category') == 'Tafseer')
            
            print(f"✅ Alhamdulillah! Islamic knowledge database ready:")
            print(f"   Total entries: {count}")
            print(f"   Estimated verses: ~{verse_count * (count // 10) if count >= 10 else verse_count}")
            print(f"   Estimated tafseer entries: ~{tafseer_count * (count // 10) if count >= 10 else tafseer_count}")
        except Exception as e:
            print(f"⚠️ Database accessible but sample query failed: {e}")
    else:
        print("⚠️ Database is empty. Run quran_loader.py and tafseer_loader.py to populate it.")

    if github_client:
        print(f"✅ GitHub Models ({GITHUB_MODEL_NAME}) is configured and ready.")
    else:
        print("❌ GitHub Models is NOT configured. Please ensure GITHUB_TOKEN is set in your .env file.")
    print("----------------------\n")

@app.get("/")
async def root():
    count = collection.count()
    
    status_msg = "Ready to serve with Quran"
    try:
        if count > 0:
            sample = collection.query(query_texts=["guidance"], n_results=min(5, count))
            has_tafseer = any(meta.get('category') == 'Tafseer' for meta in sample['metadatas'][0])
            if has_tafseer:
                status_msg += " + Tafseer"
    except:
        pass
    
    llm_status = f"GitHub Models ({GITHUB_MODEL_NAME}) loaded" if github_client else "GitHub Models NOT loaded"

    return {
        "message": "Islamic Knowledge API - Complete Quran + Tafseer Edition (GitHub Models)", 
        "status": f"{status_msg}. {llm_status}",
        "total_entries": count,
        "blessing": "Bismillah! Ready to serve the Ummah with comprehensive Islamic knowledge 🕌📚",
        "llm_provider": "GitHub Models - Powered by OpenAI GPT-4"
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
        
        # Generate comprehensive response using GitHub Models
        answer = generate_enhanced_response_with_tafseer(
            request.question, 
            relevant_texts,
            verse_count,
            tafseer_count
        )
        
        # Format sources with complete information
        sources = []
        for text in relevant_texts[:5]:  
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
            greeting="Hey buddy! 😊",
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
            cat = metadata.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
            
            if cat == 'Quran':
                rev_type = metadata.get('revelation_type', 'Unknown')
                if rev_type != 'Unknown':
                    revelation_types[rev_type] = revelation_types.get(rev_type, 0) + 1
                
                surah_name = metadata.get('surah_english_name', 'Unknown')
                if surah_name != 'Unknown':
                    surahs[surah_name] = surahs.get(surah_name, 0) + 1
            
            elif cat == 'Tafseer':
                tafseer_name = metadata.get('tafseer_name', 'Unknown')
                if tafseer_name != 'Unknown':
                    tafseers[tafseer_name] = tafseers.get(tafseer_name, 0) + 1
        
        scale_factor = total_count / sample_size if sample_size > 0 else 1
        
        estimated_stats = {
            "categories": {k: int(v * scale_factor) for k, v in categories.items()},
            "revelation_types": {k: int(v * scale_factor) for k, v in revelation_types.items()},
            "available_tafseers": list(tafseers.keys()),
            "sample_surahs": list(surahs.keys())[:10]
        }
        
        return {
            "total_entries": total_count,
            "estimated_breakdown": estimated_stats,
            "status": "Alhamdulillah - Comprehensive Islamic knowledge base ready!",
            "llm_provider": "GitHub Models - OpenAI GPT-4 Integration",
            "capabilities": [
                "Complete Quran verses with context",
                "Multiple tafseer commentaries",
                "Scholarly interpretations",
                "Cross-referenced responses",
                "AI-powered answer generation (via GitHub Models)"
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
        
        capabilities = {
            "database_accessible": True,
            "has_content": count > 0,
            "quran_loaded": False,
            "tafseer_loaded": False,
            "remote_llm_status": "not_configured"
        }
        
        if github_client:
            capabilities["remote_llm_status"] = "configured_and_ready"
        else:
            capabilities["remote_llm_status"] = "not_configured_or_failed"

        if count > 0:
            sample = collection.query(query_texts=["Allah"], n_results=min(10, count))
            
            for metadata in sample['metadatas'][0]:
                if metadata.get('category') == 'Quran':
                    capabilities["quran_loaded"] = True
                elif metadata.get('category') == 'Tafseer':
                    capabilities["tafseer_loaded"] = True
        
        health_status = "healthy" if capabilities["has_content"] and capabilities["remote_llm_status"] == "configured_and_ready" else "needs_attention"
        
        return {
            "status": health_status,
            "database_entries": count,
            "capabilities": capabilities,
            "llm_provider": "GitHub Models",
            "message": "Alhamdulillah, comprehensive Islamic knowledge system with GitHub Models is running!",
            "recommendations": [
                "Get your GitHub token from: https://github.com/settings/tokens",
                "Ensure GITHUB_TOKEN is set in your .env file.",
                "Run `quran_loader.py` if no Quran verses detected",
                "Run `tafseer_loader.py` to add scholarly commentaries",
                "Use `/ask` endpoint for comprehensive Islamic guidance"
            ] if health_status == "needs_attention" else [
                "System fully operational - ready for Islamic Q&A with GitHub Models!",
                "GitHub Models provides excellent OpenAI GPT-4 access",
                "Try asking complex questions for verse + tafseer responses"
            ]
        }
        
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/surah/{surah_number}")
async def get_surah_info(surah_number: int):
    """Get information about a specific surah"""
    try:
        results = collection.query(
            query_texts=["Allah guidance"],
            n_results=200,
            where={"surah_number": str(surah_number)}
        )
        
        if not results['documents'][0]:
            return {"error": f"No data found for Surah {surah_number}"}
        
        first_metadata = results['metadatas'][0][0]
        surah_info = {
            "surah_number": surah_number,
            "arabic_name": first_metadata.get('surah_name', 'Unknown'),
            "english_name": first_metadata.get('surah_english_name', 'Unknown'),
            "revelation_type": first_metadata.get('revelation_type', 'Unknown'),
        }
        
        verse_count = sum(1 for meta in results['metadatas'][0] if meta.get('category') == 'Quran')
        tafseer_count = sum(1 for meta in results['metadatas'][0] if meta.get('category') == 'Tafseer')
        
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