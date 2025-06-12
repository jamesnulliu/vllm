import torch
import pickle
import os
from datetime import datetime
from pathlib import Path
from vllm.distributed.parallel_state import get_tp_group


class RouterLog:
    _temp_dir = ".tmp/vllm_router_logs"

    @classmethod
    def _ensure_temp_dir(cls):
        Path(cls._temp_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def log(
        cls,
        logits_list: list[torch.Tensor],
        topk_weights_list: list[torch.Tensor],
        topk_ids_list: list[torch.Tensor],
        model_name: str,
        function_name: str,
    ):
        cls._ensure_temp_dir()

        logits = torch.stack(logits_list).cpu()
        topk_weights = torch.stack(topk_weights_list).cpu()
        topk_ids = torch.stack(topk_ids_list).cpu()

        log_entry = {
            "model_name": model_name,
            "function_name": function_name,
            "timestamp": datetime.now().isoformat(),
            "logits": logits,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "tp_world_size": getattr(get_tp_group(), "world_size", None),
            "tp_rank": getattr(get_tp_group(), "rank", None),
        }

        # Save each entry to a separate file
        filename = f"router_log_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.pkl"
        filepath = os.path.join(cls._temp_dir, filename)

        with open(filepath, "wb") as f:
            pickle.dump(log_entry, f)

    @classmethod
    def save(cls, path: str):
        cls._ensure_temp_dir()

        # Collect all log files
        log_files = [
            f for f in os.listdir(cls._temp_dir) if f.startswith("router_log_")
        ]
        output_list = []

        for log_file in sorted(log_files):
            filepath = os.path.join(cls._temp_dir, log_file)
            with open(filepath, "rb") as f:
                log_entry = pickle.load(f)
                output_list.append(log_entry)

        print(f"{len(output_list)} router logits saved to {path}")
        torch.save(output_list, path)

        # Clean up temp files
        for log_file in log_files:
            os.remove(os.path.join(cls._temp_dir, log_file))
