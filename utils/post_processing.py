"""Post-processing utilities — log parsing for resume and loss-epoch curves."""
import json
import re
from typing import Dict, List

from utils.utils import recursive_merge


class LogParser:
    """Reads a training log file to support training resumption and performance extraction.

    Args:
        log_file: Path to the training log file.
        param: ModelParams configuration (for context, e.g. feature lists).
    """

    def __init__(self, log_file: str, param=None) -> None:
        self.log_file = log_file
        self.param = param

    def get_performance(self) -> Dict:
        """Parse log and return merged performance dict (for loss-epoch plots)."""
        with open(self.log_file) as lf:
            text = lf.readlines()
        pattern = r'\{.*\}'
        dicts = []
        for line in text:
            if 'EarlyStopping' in line or 'Ending...' in line:
                break
            if '"Epoch"' not in line:
                continue
            match = re.search(pattern, line)
            if match:
                dicts.append(json.loads(match.group(0)))
        return recursive_merge(dicts)

    def get_feature(self) -> Dict:
        """Extract feature configuration from log."""
        with open(self.log_file) as lf:
            text = lf.readlines()
        param = json.loads(text[1])
        feature = {
            'node': param.get('node_attr_list', []),
            'edge': param.get('edge_attr_list', []),
            'graph': param.get('graph_attr_list', []),
        }
        return feature

    def restart(self, start_epoch: int) -> List[str]:
        """Return log lines for epochs before start_epoch (used when resuming)."""
        with open(self.log_file) as lf:
            text = lf.readlines()
        pre_log_text = []
        i = 1
        for line in text:
            if i == start_epoch:
                break
            if '"Epoch"' in line:
                pre_log_text.append(line)
                i += 1
        return pre_log_text
