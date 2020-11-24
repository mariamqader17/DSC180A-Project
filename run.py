import sys
import os
import json
sys.path.insert(0, 'src')

from all import findFeatures

def main(targets):
    '''
    this function will run the main VPN_XRAY project with the given targets, for this repo we are just focusing on the test target
    '''

    if 'test' in targets:
        with open('config/test_params.json') as fh:
            feature_cfg = json.load(fh)

        # makes the feature target
        features, labels = findFeatures(**feature_cfg)

    return 

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
