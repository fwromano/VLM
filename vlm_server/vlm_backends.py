import io
import os
import time
from typing import Any, Dict
import traceback


class ModelManager:
    def __init__(self, cfg: Dict[str, Any], env: Dict[str, Any]):
        self.cfg = cfg
        self.env = env
        self.capabilities = {
            'vllm': bool(env.get('vllm_available')) and env.get('device') == 'cuda',
        }
        self.loaded_models: Dict[str, Any] = {}
        self._init_device()
        self._lazy = {}

    def _init_device(self):
        try:
            import torch
        except Exception:
            self.device = 'cpu'
            return

        if self.env.get('device') == 'cuda':
            self.device = 'cuda'
        elif self.env.get('device') == 'mps':
            # MPS device for Apple Silicon
            self.device = 'mps'
        else:
            self.device = 'cpu'

    def _ensure_transformers(self):
        if 'transformers' in self._lazy:
            return
        from transformers import (
            Gemma3ForConditionalGeneration,
            AutoProcessor,
            AutoTokenizer,
            AutoModelForCausalLM,
        )
        import torch
        from PIL import Image
        self._lazy['transformers'] = {
            'Gemma3ForConditionalGeneration': Gemma3ForConditionalGeneration,
            'AutoProcessor': AutoProcessor,
            'AutoTokenizer': AutoTokenizer,
            'AutoModelForCausalLM': AutoModelForCausalLM,
            'torch': torch,
            'Image': Image,
        }

    def _ensure_vllm(self):
        if 'vllm' in self._lazy:
            return
        from vllm import LLM, SamplingParams
        self._lazy['vllm'] = {
            'LLM': LLM,
            'SamplingParams': SamplingParams,
        }

    def _load_transformers_model(self, model_key: str):
        self._ensure_transformers()
        T = self._lazy['transformers']
        model_cfg = self.cfg['models'][model_key]
        hf_id = model_cfg['hf_id']
        dtype = T['torch'].bfloat16 if self.device in ('cuda', 'mps') else T['torch'].float32

        vision = model_cfg['type'] == 'vision'
        if vision:
            model = T['Gemma3ForConditionalGeneration'].from_pretrained(
                hf_id,
                torch_dtype=dtype,
                device_map={"": 0} if self.device == 'cuda' else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            processor = T['AutoProcessor'].from_pretrained(hf_id, use_fast=True)
            if self.device != 'cuda':
                model = model.to(self.device)
            self.loaded_models[model_key] = {
                'backend': 'transformers',
                'type': 'vision',
                'model': model,
                'processor': processor,
            }
        else:
            # text-only fallback
            model = T['AutoModelForCausalLM'].from_pretrained(
                hf_id,
                torch_dtype=dtype,
                device_map={"": 0} if self.device == 'cuda' else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            tokenizer = T['AutoTokenizer'].from_pretrained(hf_id, use_fast=True)
            if self.device != 'cuda':
                model = model.to(self.device)
            self.loaded_models[model_key] = {
                'backend': 'transformers',
                'type': 'text',
                'model': model,
                'tokenizer': tokenizer,
            }

    def _load_vllm_model(self, model_key: str):
        self._ensure_vllm()
        V = self._lazy['vllm']
        model_cfg = self.cfg['models'][model_key]
        hf_id = model_cfg['hf_id']
        llm = V['LLM'](
            model=hf_id,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.80,
            trust_remote_code=True,
            max_model_len=4096,
        )
        sampling = V['SamplingParams'](
            temperature=0.1,
            top_p=0.9,
            max_tokens=256,
        )
        self.loaded_models[model_key] = {
            'backend': 'vllm',
            'type': self.cfg['models'][model_key]['type'],
            'llm': llm,
            'sampling': sampling,
        }

    def _get_or_load(self, model_key: str, backend: str):
        # Reuse if loaded with any backend that satisfies request
        entry = self.loaded_models.get(model_key)
        if entry and entry['backend'] == backend:
            return entry
        # Load fresh
        if backend == 'vllm':
            if not self.capabilities['vllm']:
                raise RuntimeError('vLLM not available on this system')
            self._load_vllm_model(model_key)
        else:
            self._load_transformers_model(model_key)
        return self.loaded_models[model_key]

    def _preprocess_image(self, pil_img, target_size: int) -> Any:
        # Square pad then resize for consistency
        from PIL import Image
        w, h = pil_img.size
        if w != h:
            side = max(w, h)
            bg = Image.new('RGB', (side, side), (0, 0, 0))
            bg.paste(pil_img, ((side - w) // 2, (side - h) // 2))
            pil_img = bg
        if target_size and pil_img.size[0] != target_size:
            pil_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        return pil_img

    def analyze_image(self, pil_img, question: str, model_key: str, backend: str, fast: bool = False) -> Dict[str, Any]:
        if model_key not in self.cfg['models']:
            raise ValueError(f'Unknown model: {model_key}')
        entry = self._get_or_load(model_key, backend)
        mcfg = self.cfg['models'][model_key]
        target_size = 512 if fast else mcfg.get('input_size', 896)
        pil_img = self._preprocess_image(pil_img, target_size)

        t0 = time.time()
        if backend == 'transformers':
            T = self._lazy['transformers']
            model = entry['model']
            processor = entry['processor']
            messages = [{
                'role': 'user',
                'content': [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": question},
                ],
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, images=[pil_img], return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # Try primary generation settings, then fall back to safer settings on error
            gen_kwargs = dict(
                max_new_tokens=200 if not fast else 120,
                do_sample=False,
                use_cache=True,
            )
            if getattr(processor, 'tokenizer', None) and getattr(processor.tokenizer, 'eos_token_id', None) is not None:
                gen_kwargs['pad_token_id'] = processor.tokenizer.eos_token_id
            else:
                # Fallback to model config eos if tokenizer missing pad
                if getattr(model.config, 'eos_token_id', None) is not None:
                    gen_kwargs['pad_token_id'] = model.config.eos_token_id

            try:
                with T['torch'].no_grad():
                    out = model.generate(
                        **inputs,
                        **gen_kwargs,
                    )
            except Exception:
                # Safer fallback: disable cache, reduce tokens
                with T['torch'].no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=100 if not fast else 80,
                        do_sample=False,
                        use_cache=False,
                    )
            try:
                input_ids = inputs['input_ids'][0]
                output_ids = out[0]
                new_ids = output_ids[len(input_ids):]
                text_out = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            except Exception:
                # Fallback decoding path
                try:
                    text_out = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
                except Exception:
                    # Last resort: plain tokenizer decode
                    tok = getattr(processor, 'tokenizer', None)
                    if tok is not None:
                        text_out = tok.decode(out[0], skip_special_tokens=True)
                    else:
                        raise
            return {
                'text': text_out,
                'backend': 'transformers',
                'model': model_key,
                'timings': {'generate': time.time() - t0},
            }
        else:
            # vLLM path: multimodal may be experimental; fall back to text-only prompt
            V = self._lazy['vllm']
            llm = entry['llm']
            sampling = entry['sampling']
            prompt = f"<image>\nUser: {question}\nAssistant:"
            try:
                from vllm import MultiModalData
                mm = MultiModalData(type='image', data=pil_img)
                outputs = llm.generate(
                    prompts=[prompt],
                    sampling_params=sampling,
                    multi_modal_data=[mm],
                )
                text_out = outputs[0].outputs[0].text.strip()
            except Exception:
                # Fallback to text-only
                outputs = llm.generate([f"User: {question}\nAssistant:"], sampling)
                text_out = "[Text-only mode] " + outputs[0].outputs[0].text.strip()
            return {
                'text': text_out,
                'backend': 'vllm',
                'model': model_key,
                'timings': {'generate': time.time() - t0},
            }

    def answer_over_text(self, context: str, question: str, model_key: str, backend: str) -> str:
        # Simple RAG prompt; use text-only path for both backends
        if backend == 'vllm':
            self._ensure_vllm()
            V = self._lazy['vllm']
            entry = self._get_or_load(model_key, 'vllm')
            llm = entry['llm']
            sampling = entry['sampling']
            prompt = (
                "You are an assistant that answers questions about the given transcript.\n"
                "Transcript:\n" + context[:8000] + "\n\nQuestion: " + question + "\nAnswer:"
            )
            outputs = llm.generate([prompt], sampling)
            return outputs[0].outputs[0].text.strip()
        else:
            self._ensure_transformers()
            T = self._lazy['transformers']
            # Load a text-only instance of the same model id
            if model_key not in self.loaded_models or self.loaded_models[model_key]['type'] != 'text':
                # Attempt to reuse hf_id with AutoModelForCausalLM
                model_cfg = self.cfg['models'][model_key]
                hf_id = model_cfg['hf_id']
                dtype = T['torch'].bfloat16 if self.device in ('cuda', 'mps') else T['torch'].float32
                model = T['AutoModelForCausalLM'].from_pretrained(
                    hf_id,
                    torch_dtype=dtype,
                    device_map={"": 0} if self.device == 'cuda' else None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                tok = T['AutoTokenizer'].from_pretrained(hf_id, use_fast=True)
                if self.device != 'cuda':
                    model = model.to(self.device)
                self.loaded_models[model_key] = {
                    'backend': 'transformers', 'type': 'text', 'model': model, 'tokenizer': tok
                }
            entry = self.loaded_models[model_key]
            tok = entry['tokenizer']
            model = entry['model']
            prompt = (
                "You are an assistant that answers questions about the given transcript.\n"
                "Transcript:\n" + context[:8000] + "\n\nQuestion: " + question + "\nAnswer:"
            )
            ids = tok(prompt, return_tensors='pt').to(self.device)
            with T['torch'].no_grad():
                out = model.generate(**ids, max_new_tokens=256, do_sample=False)
            ans = tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)
            return ans.strip()

    def transcribe_mp3_bytes(self, data: bytes):
        # Use faster-whisper if available
        try:
            from faster_whisper import WhisperModel
        except Exception:
            raise RuntimeError('faster-whisper not available; install to use audio transcription')

        import tempfile
        import soundfile as sf
        import numpy as np

        # Write bytes to temp file (faster-whisper expects a path)
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tmp.write(data)
            path = tmp.name

        # Pick device
        device = 'cuda' if self.env.get('device') == 'cuda' else 'cpu'
        compute_type = 'float16' if device == 'cuda' else 'int8'
        model = WhisperModel('small', device=device, compute_type=compute_type)
        t0 = time.time()
        segments, info = model.transcribe(path, vad_filter=True)
        text = ''.join([seg.text for seg in segments]).strip()
        return text, {'duration': getattr(info, 'duration', None), 'transcribe_time': time.time() - t0}
