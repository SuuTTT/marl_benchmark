import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class FrozenSteeringPipeline:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading {model_name} on {self.device}...")

        # 1. Load Frozen Model in 4-bit to save VRAM for Activations (Benchmark Setting)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.model.eval()
        
        # Internal storage for hooks
        self._activations = {}
        self._steering_vector = None
        self._target_layer = None
        self._hooks = []

    def _get_layer(self, layer_idx):
        # Access specific layer (Architecture dependent - this works for Llama/Mistral)
        return self.model.model.layers[layer_idx]

    # =====================================================
    # PHASE 1: READING (Representation Extraction)
    # =====================================================
    def extract_activations(self, prompt, layer_idx):
        """Runs a forward pass and captures the hidden state at a specific layer."""
        self._activations = {}
        
        def hook_fn(module, input, output):
            # output[0] is the hidden state tensor (Batch, Seq, Dim)
            # We grab the last token's embedding
            self._activations['last_token'] = output[0][:, -1, :].detach().cpu()

        # Register hook
        layer = self._get_layer(layer_idx)
        handle = layer.register_forward_hook(hook_fn)
        
        # Run Inference
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model(**inputs)
        
        # Cleanup
        handle.remove()
        return self._activations['last_token']

    def create_steering_vector(self, positive_prompt, negative_prompt, layer_idx):
        """Creates a concept vector by subtracting Negative from Positive."""
        print(f"🧪 Extracting concept at Layer {layer_idx}...")
        pos_vec = self.extract_activations(positive_prompt, layer_idx)
        neg_vec = self.extract_activations(negative_prompt, layer_idx)
        
        # Simple Difference Vector (The core of Representation Engineering)
        self._steering_vector = (pos_vec - neg_vec).to(self.device)
        self._target_layer = layer_idx
        return self._steering_vector

    # =====================================================
    # PHASE 2: WRITING (Steering / Intervention)
    # =====================================================
    def generate_steered(self, prompt, coefficient=1.0, max_new_tokens=50):
        """Generates text while adding the steering vector to the hidden state."""
        
        def steering_hook(module, input, output):
            # output[0] shape: (Batch, Seq, Hidden_Dim)
            # We add the vector to all tokens (or just the last one)
            if self._steering_vector is not None:
                # Broadcast extraction vector to match sequence length
                output[0][:, :, :] += coefficient * self._steering_vector
            return output

        # Register hook
        layer = self._get_layer(self._target_layer)
        handle = layer.register_forward_hook(steering_hook)
        
        print(f"\n🧠 Generating with Steering Coeff: {coefficient}")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        handle.remove() # Critical: Remove hook or it will stay forever!
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# =====================================================
# 3. THE EXPERIMENT
# =====================================================
if __name__ == "__main__":
    # Use Mistral-7B as originally suggested in context_llm.md
    # Note: Ensure you have accepted terms on HF Hub or use a smaller public model if needed.
    pipeline = FrozenSteeringPipeline("mistralai/Mistral-7B-Instruct-v0.3")

    # A. Define a "Concept" (e.g., Usefulness vs Refusal)
    pos_prompt = "I cannot answer that because it is illegal."
    neg_prompt = "Sure! Here is the answer to your question."
    
    # B. Extract the vector from Layer 15
    vector = pipeline.create_steering_vector(pos_prompt, neg_prompt, layer_idx=15)

    # C. Test Query
    test_query = "How do I make a cup of tea?"
    
    # Baseline (Normal)
    print(f"\n--- Baseline Response ---")
    print(pipeline.generate_steered(test_query, coefficient=0.0))

    # Steered (Refusal Mode)
    print(f"\n--- Steered Response (Refusal Vector) ---")
    print(pipeline.generate_steered(test_query, coefficient=2.0)) 
    
    # Steered (Opposite - Super Helpful Mode)
    print(f"\n--- Steered Response (Helpful Vector) ---")
    print(pipeline.generate_steered(test_query, coefficient=-2.0))
