import os
import json
import pickle
from tqdm import tqdm
from loguru import logger
from typing import Optional, List, Dict, Any
from common import (
    format_state,
    format_tactic,
)
def load_data_from_json(_data: str, normalize_tactics: bool) -> List[Example]:
    data = []
    i = 0
    for thm in tqdm(_data):
        i += 1
        for tac in thm["traced_tactics"]:
            if "annotated_tactic" in tac:
                tactic = format_tactic(*tac["annotated_tactic"], normalize_tactics)
            else:
                tactic = format_tactic(tac["tactic"], [], normalize_tactics)

            data.append(
                {
                    "url": thm["url"],
                    "commit": thm["commit"],
                    "file_path": thm["file_path"],
                    "full_name": thm["full_name"],
                    "state_before": format_state(tac["state_before"]),
                    "tactic": tactic,
                    "state_after": format_state(tac["state_after"])

                }
            )

    logger.info(f"{len(data)} examples loaded")
    print(f"the number of theorem is {i}")
    return data

if __name__ == '__main__':
    #data = load_data('/Users/anchenyang/Desktop/cm/LeanDojo_Data_Original/leandojo_benchmark_4/random/val.json', True)
    data = load_data('/Users/anchenyang/Desktop/Lean4Example.json', True)
    print(data[0])


