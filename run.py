import sys
import os
import json
sys.path.insert(0, 'src')

from all import findFeatures

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    
    `main` runs the targets in order of data=>analysis=>model.
    '''

    if 'test' in targets:
        with open('config/test_params.json') as fh:
            feature_cfg = json.load(fh)

        # make the data target
        features, labels = findFeatures(**feature_cfg)

    return 

if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)
