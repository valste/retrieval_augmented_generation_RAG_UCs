import os

def get_openAI_apiKey():
    """
    Retrieves the openAI API key from file.
    """
    
    input(""""Please create 'your.openai.api.key' containing your OpenAI API key.
              Keep in mind to exclude your key file from commiting to public repositories!!!
              Press Enter to continue...   """)
    
    api_key_file = None
    
    if os.path.exists("vst.openai.api.key"):
        api_key_file = "vst.openai.api.key"
    elif os.path.exists("your.openai.api.key"):
        api_key_file = "your.openai.api.key"
    else:
        raise FileNotFoundError("""No API key file found. Please create 'your.openai.api.key' containing your OpenAI API key.
                                Keep in mind to exclude your key file from commiting to public repositories!!!""")
    
    try:
        # --- Read API key 
        with open(api_key_file, "r", encoding="utf-8") as f:
            key = f.read().strip()
            return key
                
    except Exception as e:
        print(f"Could not read API key file: {type(e).__name__}: {e}")
        

def set_apiKey_env():
    """
    Sets the OPENAI_API_KEY environment variable.
    """
    api_key = get_openAI_apiKey()
    os.environ["OPENAI_API_KEY"] = api_key