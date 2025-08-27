# model.py
import os, torch, logging, gc
import tiktoken
from typing import Optional, List, Dict
from GPT import GPTModel
from GPTspam import GPTModel as GPTModelSpam
from logging_utils import log_memory_usage

TORCH_THREADS = int(os.getenv("TORCH_THREADS", "1"))
MODEL_NAME    = os.getenv("MODEL_NAME", "gpt2")   # base config/checkpoint name
CKPT_PATH     = os.getenv("CKPT_PATH", "model.pth")  # your .pth file path
MAX_CTX       = int(os.getenv("MAX_CTX", "512"))  # limit to keep cheap/fast

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.05,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"






class GPT2Service:
    def __init__(self, task_name):
        logger.debug("Initializing GPT2Service with task_name=%s", task_name)
        try:
            torch.set_num_threads(TORCH_THREADS)
            logger.debug("Set TORCH_THREADS=%s", TORCH_THREADS)
            self.task_name = task_name
            self.device = torch.device("cpu")
            logger.debug("Set device to CPU")

            self.tokenizer = tiktoken.get_encoding(MODEL_NAME)

            if task_name == "spam":
                CHOOSE_MODEL = "gpt2-small (124M)"
                BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
                num_classes = 2
                self.model = GPTModelSpam(BASE_CONFIG)
                self.model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
            else:
                CHOOSE_MODEL = "gpt2-medium (355M)"
                BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
                self.model = GPTModel(BASE_CONFIG)

            if task_name == "spam":
                self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "spam_model.pth"), map_location=self.device))
            else:
                self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), CKPT_PATH), map_location=self.device))

            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded for task_name=%s", task_name)
            log_memory_usage(f"model_loaded_{task_name}")
        except Exception:
            logger.error("Failed to initialize GPT2Service for task_name=%s", task_name, exc_info=True)
            raise

    def cleanup(self):
        logger.debug("Entering cleanup for task_name=%s", self.task_name)
        try:
            if hasattr(self, "model"):
                logger.debug("Deleting model for task_name=%s", self.task_name)
                self.model.to("cpu")
                del self.model
            if hasattr(self, "tokenizer"):
                logger.debug("Deleting tokenizer for task_name=%s", self.task_name)
                del self.tokenizer
            gc.collect()
            logger.debug("Garbage collection invoked during cleanup")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared during cleanup")
            logger.info("Cleanup complete for task_name=%s", self.task_name)
        except Exception:
            logger.error("Error during cleanup for task_name=%s", self.task_name, exc_info=True)
            raise


    def generate_text_simple(self, idx, max_new_tokens, context_size, temperature=1.2, top_k=3, eos_id=None):
        '''
        iterate for max_new_tokens times
        get the last context_size tokens of each sentence
        inference
        get last vector of each sentence output
        topk: set others to -inf
        temp: divide=>softmax=>moltinomial
        if eos: stop generating
        else: append to idx with cat

        '''

        # For-loop is the same as before: Get logits, and only focus on last time step
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = self.model(idx_cond)
            logits = logits[:, -1, :]

            # New: Filter logits with top_k sampling
            if top_k is not None:
                # Keep only top_k values
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

            # New: Apply temperature scaling
            if temperature > 0.0:
                logits = logits / temperature

                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            # Otherwise same as before: get idx of the vocab entry with the highest logits value
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

            if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                break

            # Same as before: append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

        return idx



    def text_to_token_ids(self, text):
        encoded = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
        return encoded_tensor

    def token_ids_to_text(self, token_ids):
        flat = token_ids.squeeze(0) # remove batch dimension
        return self.tokenizer.decode(flat.tolist())
    
    def generate_and_print_sample(self, start_context):
        self.model.eval()
        context_size = self.model.pos_emb.weight.shape[0]
        encoded = self.text_to_token_ids(start_context).to(self.device)
        with torch.no_grad():
            token_ids = self.generate_text_simple(
                idx=encoded,
                max_new_tokens=50, context_size=context_size
            )
        decoded_text = self.token_ids_to_text(token_ids)
        return decoded_text
    

    def classify_review(self, text, pad_token_id=50256):
        self.model.eval()

        # Prepare inputs to the model
        input_ids = self.tokenizer.encode(text)
        supported_context_length = self.model.pos_emb.weight.shape[0]
        # Note: In the book, this was originally written as pos_emb.weight.shape[1] by mistake
        # It didn't break the code but would have caused unnecessary truncation (to 768 instead of 1024)

        # Truncate sequences if they too long
        input_ids = input_ids[:supported_context_length]

        # Pad sequences to the longest sequence
        input_ids += [pad_token_id] * (supported_context_length - len(input_ids))
        input_tensor = torch.tensor(input_ids, device=self.device).unsqueeze(0) # add batch dimension


        # Model inference
        with torch.no_grad():
            logits = self.model(input_tensor)[:, -1, :]  # Logits of the last output token
        predicted_label = torch.argmax(logits, dim=-1).item()

        # Return the classified result
        return "spam" if predicted_label == 1 else "not spam"

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        stop: Optional[List[str]] = None
    ) -> str:
        if self.task_name == "spam":
            return self.classify_review(prompt)
        else:
            return self.generate_and_print_sample("###### question:"+prompt+"\n\n###### Response:")



# singleton (lazy) per task
logger = logging.getLogger(__name__)
_services: Dict[str, GPT2Service] = {}


def clear_services() -> None:
    logger.debug("Entering clear_services")
    log_memory_usage("clear_services_start")
    try:
        if _services:
            logger.debug("Services to clear: %d", len(_services))
            for task in list(_services.keys()):
                logger.info("Clearing service for task_name=%s", task)
                service = _services.pop(task)
                logger.debug("Invoking cleanup for task_name=%s", task)
                service.cleanup()
                del service
            gc.collect()
            logger.debug("Garbage collection invoked")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared")
            logger.info("All services cleared")
            log_memory_usage("clear_services_end")
        else:
            logger.debug("No services to clear")
    except Exception:
        logger.error("Failed to clear services", exc_info=True)
        raise


def get_service(task_name: str) -> GPT2Service:
    logger.debug("Entering get_service with task_name=%s", task_name)
    log_memory_usage(f"get_service_start_{task_name}")
    try:
        if task_name not in _services:
            logger.info("Clearing existing services before loading task_name=%s", task_name)
            clear_services()
            logger.info("Creating new GPT2Service for task_name=%s", task_name)
            _services[task_name] = GPT2Service(task_name)
            log_memory_usage(f"service_created_{task_name}")
            logger.debug("Service created for task_name=%s", task_name)
        else:
            logger.debug("Reusing existing GPT2Service for task_name=%s", task_name)
        logger.debug("Returning service for task_name=%s", task_name)
        return _services[task_name]
    except Exception:
        logger.error("Error retrieving service for task_name=%s", task_name, exc_info=True)
        raise
