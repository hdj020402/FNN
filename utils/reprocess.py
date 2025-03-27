import yaml, os
from typing import Dict

KEYS = [
    'sdf_file',
    'data_file',
    'weight_file',
    'default_feature',
    'feature_list',
    'target_list',
    'target_transform',
]

def reprocess(param: Dict) -> bool:
    try:
        with open(os.path.join(param['path'], 'processed/model_parameters.yml'), 'r', encoding = 'utf-8') as mp:
            param_pre: Dict = yaml.full_load(mp)
        sub_dict1 = {key: param[key] for key in KEYS if key in param}
        sub_dict2 = {key: param_pre[key] for key in KEYS if key in param_pre}
        return not sub_dict1 == sub_dict2
    except Exception:
        return True

