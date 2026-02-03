
import os
import json
import uuid
from offline_api import OfflineAPI
from config import APIConfig, StoreConfig

def main():
    # 1. Configuration
    # You can override defaults here or via environment variables
    # e.g. os.environ["LLM_BASE_URL"] = "..."
    # For this example, we assume defaults or env vars are set correctly.
    
    # We can pass custom configs if needed
    api_cfg = APIConfig()
    store_cfg = StoreConfig() 
    
    # Initialize the Offline API
    print("Initializing OfflineAPI...")
    offline_api = OfflineAPI(api_cfg=api_cfg, store_cfg=store_cfg)

    # 2. Add History Example
    # Let's verify we can add some past conversations.
    # Imagine a user 'alice' had a conversation 'conv_1' about her favorite food.
    
    user_id = "alice"
    conv_id = "conv_1"
    
    history_messages = [
        {"role": "user", "content": "Hi, I'm Alice. I love pizza."},
        {"role": "assistant", "content": "Hello Alice! It's nice to meet you. Pizza is delicious."},
        {"role": "user", "content": "My favorite topping is pepperoni."},
        {"role": "assistant", "content": "Pepperoni is a classic choice!"}
    ]
    
    print(f"\nAdding history for user='{user_id}', conv_id='{conv_id}'...")
    added_ids = offline_api.add_history(user_id=user_id, conv_id=conv_id, history_messages=history_messages)
    print(f"Added {len(added_ids)} turns to memory. IDs: {added_ids}")

    # 3. Chat Example
    # Now let's start a new conversation 'conv_2' and ask about the food.
    # The system should retrieve the memory from 'conv_1'.
    
    new_conv_id = "conv_2"
    query = "Do you remember what my favorite food is?"
    
    print(f"\nChatting with user='{user_id}', conv_id='{new_conv_id}'...")
    print(f"Query: {query}")
    
    # We use a mocked print here effectively, realistically this would call the LLM.
    # For the sake of this example running without a real LLM server, 
    # the code will attempt to connect. If no server is running, it might fail or hang.
    # The user asked for the code, so we provide the valid python code to run.
    
    try:
        response = offline_api.chat(
            user_id=user_id, 
            conv_id=new_conv_id, 
            query=query
        )
        
        print("\n=== Response ===")
        print(f"Output: {response['choices'][0]['message']['content']}")
        
        print("\n=== Retrieved Memory Debug Info ===")
        # Check if we retrieved the pizza facts
        replay = response['retrieved']['replay']
        for item in replay:
            print(f"[Replay] {item['ts']} - {item['raw_user']} -> {item['raw_assistant']}")
            
    except Exception as e:
        print(f"\n[Error during chat]: {e}")
        print("Note: Ensure your VLLM and Embedding servers are running at the URLs defined in config.py")

if __name__ == "__main__":
    main()
