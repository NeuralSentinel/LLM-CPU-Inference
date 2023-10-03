from langchain.llms import CTransformers
from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer

#llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGML", model_type='llama', gpu_layers=50, model_file='llama-2-7b-chat.ggmlv3.q8_0.bin')  # Load model from GGML model repo.
#tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-Chat-GGML")  # Load tokenizer from original model repo.

#"C:\Users\somii\.cache\huggingface\hub\models--TheBloke--Llama-2-7B-Chat-GGML\snapshots\b616819cd4777514e3a2d9b8be69824aca8f5daf\llama-2-7b-chat.ggmlv3.q8_0.bin"
# Local CTransformers wrapper for Llama-2-7B-Chat
llm = CTransformers(model='TheBloke/Llama-2-7B-Chat-GGML', # Location of downloaded GGML model
                    model_file='llama-2-7b-chat.ggmlv3.q8_0.bin',
                    model_type='llama', # Model type Llama
                    config={'max_new_tokens': 256,
                            'temperature': 0.01},
                   gpu_layers=50)