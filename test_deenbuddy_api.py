
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test if the API is running"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print("Health Check:")
        print(json.dumps(response.json(), indent=2))
        print("-" * 50)
    except Exception as e:
        print(f"Error: {e}")

def ask_question(question):
    """Ask a question to the Islamic chatbot"""
    try:
        payload = {"question": question}
        response = requests.post(f"{BASE_URL}/ask", json=payload)
        
        print(f"Question: {question}")
        print("Response:")
        result = response.json()
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error asking question: {e}")

if __name__ == "__main__":
    print("Testing Islamic Knowledge API")
    print("=" * 50)
    
    # Test health
    test_health()
    
    # Test questions
    test_questions = [
        "quran or hadith rules on sisters haqooq",
    ]
    
    for question in test_questions:
        ask_question(question)