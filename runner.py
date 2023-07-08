from train import main
import yaml
from itertools import product
import pdb
import traceback
if __name__ == '__main__':
    runner_opt = yaml.safe_load(open(f'configs/runner.yaml', 'r'))
    # runner_opt = {k: v for k, v in runner_opt.items() if k!='model'}
    keys = [[k]*len(v) for k , v in runner_opt.items()]
    values = list(runner_opt.values())
    for kt, vt in zip(product(*keys), product(*values)):
        item_dict = {k:v for k , v in zip(kt, vt)}
        opt = yaml.safe_load(open(f"configs/train_cfg/{item_dict['model']}.yaml", 'r'))
        for k ,v in item_dict.items():
            if k!= 'model':
                flag = False
                for mk in ['training', 'model', 'dataset']:
                    if k in opt[mk]:
                        opt[mk][k] = v
                        flag = True
                if not flag:
                    print('key not found')
                    exit(1)
        try:
            with open(f"configs/train_cfg/{item_dict['model']}.yaml", 'w') as f:
                yaml.dump(opt, f)
            print('running item_dict', item_dict, '\n\n')
            main(f"configs/train_cfg/{item_dict['model']}.yaml")
        except Exception as e:
            print(traceback.format_exc())
            continue