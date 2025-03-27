from typing import Dict, Callable

def attr_filter(main: Callable[[Dict], None], param: Dict):
    feature_list = param['feature_list']
    for i, attr in enumerate(feature_list):
        param['feature_list'] = [x for x in feature_list if x != attr]
        main(param)
