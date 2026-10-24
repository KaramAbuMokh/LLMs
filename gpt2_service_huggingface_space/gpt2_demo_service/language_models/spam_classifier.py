# Email spam check: the fine-tuned GPT-2 small model answers "spam" or "not spam"
import tiktoken  # GPT-2 tokenizer
import torch  # tensor library
from gpt2_demo_service.language_models.gpt_model import GPT2_SMALL_SIZE, GPTModel, build_config  # network and its size settings

PAD_TOKEN_ID = 50256  # GPT-2 end-of-text token, used to pad short emails
SPAM_CLASS_INDEX = 1  # the model's second output means spam
TRAINING_SEQUENCE_LENGTH = 120  # length emails were padded to in training; padding to 1024 made it answer "spam" every time


class SpamClassifier:  # holds the loaded spam model and labels emails
    def __init__(self, weights_path):  # weights_path points at spam_model.pth
        self.tokenizer = tiktoken.get_encoding("gpt2")  # turns text into token ids
        config = build_config(GPT2_SMALL_SIZE)  # size settings of the small model
        self.model = GPTModel(config)  # empty network of the right size
        self.model.out_head = torch.nn.Linear(config["embedding_dim"], 2)  # two outputs: not spam, spam
        saved_weights = torch.load(weights_path, map_location="cpu")  # trained weights from disk
        renamed_weights = {  # this file called the feed-forward layers "layers", the shared network says "layer"
            name.replace(".ff.layers.", ".ff.layer."): tensor for name, tensor in saved_weights.items()
        }
        self.model.load_state_dict(renamed_weights)  # fill the network with the trained weights
        self.model.eval()  # inference mode, no dropout

    @torch.inference_mode()
    def label_email(self, email_text):  # returns "spam" or "not spam"
        token_ids = self.tokenizer.encode(email_text, disallowed_special=())[:TRAINING_SEQUENCE_LENGTH]  # cut long emails
        token_ids += [PAD_TOKEN_ID] * (TRAINING_SEQUENCE_LENGTH - len(token_ids))  # pad to the training length
        logits = self.model(torch.tensor(token_ids).unsqueeze(0))[:, -1, :]  # scores read at the last position
        predicted_class = torch.argmax(logits, dim=-1).item()  # index of the higher score
        return "spam" if predicted_class == SPAM_CLASS_INDEX else "not spam"  # human-readable label
