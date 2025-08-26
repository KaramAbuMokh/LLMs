#!/usr/bin/env python3
"""
Simple test script for the GPT-2 Service API
Run this after starting the service to verify it's working
"""

import requests

BASE_URL = "http://localhost:8080"

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health check: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to service. Is it running?")
        return False

def test_generate():
    """Test the generate endpoint"""
    payload = {
        "prompt": "This is a test message for spam classification",
        "task_name": "spam",
        "max_new_tokens": 50,
        "temperature": 1.5,
        "top_p": 0.9,
        "top_k": 50
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/generate",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        print(f"Generate request: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Generated text: {result['completion']}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to service")
        return False

def test_got_generation():
    """Test GOT text generation task"""
    payload = {"prompt": "what did tyrion lannister tell sansa stark at the wedding night when they were alone and when she told him she dont want to bed him at all?", "max_new_tokens": 50, "task_name": "got"}
    
    try:
        response = requests.post(
            f"{BASE_URL}/generate",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✅ GOT generation working: {result['completion']}")
            return True
        else:
            print(f"❌ GOT generation failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to service")
        return False

def main():
    print("🧪 Testing GPT-2 Service API...")
    print("=" * 40)
    
    # Test health endpoint
    if not test_health():
        print("\n❌ Service is not running or not accessible")
        print("Make sure to start the service first:")
        print("  python app.py")
        print("  OR")
        print("  ./start.sh")
        return
    
    print("\n✅ Service is running!")
    
    # Test generate endpoint
    print("\n📝 Testing text generation...")
    if test_generate():
        print("✅ Text generation working!")
    else:
        print("❌ Text generation failed")
    
    # Test GOT generation
    print("\n📚 Testing GOT text generation...")
    test_got_generation()
    
    print("\n" + "=" * 40)
    print("🎉 Testing complete!")

if __name__ == "__main__":
    main() 