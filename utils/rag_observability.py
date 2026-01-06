import time
import uuid
import json
import functools
import threading
from datetime import datetime
from typing import List, Any, Optional

LOG_FILE = "rag_observability.jsonl"

def _append_to_log(data: dict):
    """Escreve no disco de forma atômica (simulada)."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

class RAGTracker:
    _local = threading.local()

    @classmethod
    def start_trace(cls, model_name: str = "unknown"):
        """Inicia um novo contexto de rastreamento."""
        cls._local.trace_data = {
            "trace_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "inputs": {},
            "rag_context": [],
            "prompt_snapshot": None,
            "output": None,
            "metrics": {"latency_ms": 0}
        }
        cls._local.start_time = time.time()

    @classmethod
    def log_input(cls, key: str, value: Any):
        """Registra inputs arbitrários (ex: query do usuário, configs)."""
        if hasattr(cls._local, 'trace_data'):
            cls._local.trace_data["inputs"][key] = str(value)

    @classmethod
    def log_retrieval(cls, content: str, source: str = "unknown", score: float = 0.0):
        """Registra um pedaço de texto recuperado pelo RAG."""
        if hasattr(cls._local, 'trace_data'):
            cls._local.trace_data["rag_context"].append({
                "content": str(content),
                "source": str(source),
                "score": score
            })

    @classmethod
    def log_prompt(cls, prompt_text: str):
        """Tira um snapshot do prompt final enviado ao modelo."""
        if hasattr(cls._local, 'trace_data'):
            cls._local.trace_data["prompt_snapshot"] = str(prompt_text)

    @classmethod
    def end_trace(cls, output: Any, error: Optional[str] = None):
        """Finaliza o trace, calcula latência e salva."""
        if not hasattr(cls._local, 'trace_data'):
            return

        end_time = time.time()
        latency = round((end_time - cls._local.start_time) * 1000, 2)
        
        data = cls._local.trace_data
        data["metrics"]["latency_ms"] = latency
        data["output"] = str(output)
        data["status"] = "error" if error else "success"
        data["error_message"] = str(error) if error else None

        _append_to_log(data)
        del cls._local.trace_data


def monitor_trace(model_name="default_model"):
    """
    Decorador que envolve a execução principal.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            RAGTracker.start_trace(model_name=model_name)

            try:
                if args: RAGTracker.log_input("arg_0", args[0])
                for k, v in kwargs.items():
                    RAGTracker.log_input(k, v)
            except:
                pass

            try:
                result = func(*args, **kwargs)
                RAGTracker.end_trace(output=result)
                return result
            except Exception as e:
                RAGTracker.end_trace(output=None, error=str(e))
                raise e
        return wrapper
    return decorator
