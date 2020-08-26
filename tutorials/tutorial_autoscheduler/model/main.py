from os import environ
import sys, json

from hier_lstm import Model_hier_LSTM
from json_to_tensor import *

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning) 
warnings.filterwarnings('ignore', category=UserWarning)

model_path = '/data/tiramisu/tutorials/tutorial_autoscheduler/model/hier_LSTM_fusion_tree_tagLo_transfer_5bl.pkl'

with torch.no_grad():
    device = 'cpu'
    torch.device('cpu')
    
    environ['layers'] = '600 350 200 180'
    environ['dropouts'] = '0.225 ' * 4
    
    input_size = 1267 * 2
    output_size = 1
    
    layers_sizes = list(map(int, environ.get('layers', '300 200 120 80 30').split()))
    drops = list(map(float, environ.get('dropouts', '0.2 0.2 0.1 0.1 0.1').split()))

    model = Model_hier_LSTM(input_size, output_size, hidden_sizes=layers_sizes, drops=drops)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    model.eval()

    try:
        while True:
            prog_json = input()
            sched_json = input()

            prog_json = json.loads(prog_json)
            sched_json = json.loads(sched_json)

            tree_tensor = get_representation(prog_json, sched_json)
            
            speedup = model.forward(tree_tensor)
            print(float(speedup.item()))
            
    except EOFError:
        exit()
        
