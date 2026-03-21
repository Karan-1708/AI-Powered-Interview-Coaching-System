import ollama
import time
from src.utils.diagnostics import get_logger

logger = get_logger()

class LLMClient:
    # --- DYNAMIC MODEL ROUTING ---
    MODEL_MAP = {
        "Eco": "llama3.2",            # 3B params: Runs beautifully on 8GB RAM laptops
        "Balanced": "llama3.1",       # 8B params: The standard, great for 16GB RAM
        "Pro": "mistral-nemo"         # 12B params: Heavy, highly nuanced, for 32GB+ RAM
    }

    def __init__(self, tier_string="Balanced"):
        """
        Initializes the Ollama connection and dynamically selects/pulls 
        the best model based on the user's hardware tier.
        """
        self.model_name = self._determine_model(tier_string)
        self._verify_and_pull()

    def _determine_model(self, tier_string):
        """Parses the Streamlit dropdown string to find the base tier."""
        for key in self.MODEL_MAP.keys():
            if key in tier_string:
                return self.MODEL_MAP[key]
        return self.MODEL_MAP["Balanced"] # Default fallback

    def _verify_and_pull(self):
        """Checks if the model is downloaded. If not, pulls it automatically."""
        try:
            # Fetch the list from the local server
            list_response = ollama.list()
            
            # Safely parse the response (handles both old Dicts and new Pydantic Objects)
            if hasattr(list_response, 'models'):
                installed_models = [m.model for m in list_response.models]
            elif isinstance(list_response, dict):
                installed_models = [m.get('name', m.get('model', '')) for m in list_response.get('models', [])]
            else:
                installed_models = []
            
            # Check if our target model is in the list
            if not any(self.model_name in m for m in installed_models):
                logger.warning(f"Model '{self.model_name}' not found locally. Initiating auto-pull...")
                ollama.pull(self.model_name)
                logger.info(f"Successfully downloaded {self.model_name}!")
            else:
                logger.info(f"Verified {self.model_name} is ready for inference.")
                
        except Exception as e:
            logger.error(f"Failed to connect to Ollama. Actual Error: {e}")
            raise ConnectionError(f"Ollama connection failed. Detail: {str(e)}")

    def generate_response(self, system_prompt, user_message, chat_history=None):
        """Sends the context and current answer to the LLM and retrieves the response."""
        if chat_history is None:
            chat_history = []

        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in chat_history:
            messages.append(msg)
            
        messages.append({"role": "user", "content": user_message})

        try:
            start_time = time.time()
            response = ollama.chat(model=self.model_name, messages=messages)
            duration = time.time() - start_time
            logger.info(f"Generated response using {self.model_name} in {duration:.2f}s")
            
            # Safely extract the text content
            if hasattr(response, 'message'):
                return response.message.content
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"LLM Generation Error: {e}")
            return f"System Error: Unable to generate response from AI Coach. Detail: {e}"

if __name__ == "__main__":
    # --- QUICK DIAGNOSTIC TEST ---
    print("🔌 Initiating dynamic LLM connection test...")
    try:
        client = LLMClient(tier_string="Eco")
        print(f"\n🧠 Sending test prompt to {client.model_name}...")
        
        test_system = "You are a strict technical lead. Respond in exactly one short sentence."
        test_user = "Hello, I am ready for the interview."
        
        reply = client.generate_response(system_prompt=test_system, user_message=test_user)
        print(f"\n🤖 AI Interviewer: {reply}\n")
        print("✅ Handshake successful!")
    except Exception as e:
        print(f"\n❌ Handshake failed: {e}")