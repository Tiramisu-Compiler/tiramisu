from os import environ
import json

from hier_lstm import Model_Recursive_LSTM_v2
from json_to_tensor import *

environ["MKL_THREADING_LAYER"] = "GNU"

model_path = (
    "/path/to/model/weights"
)

MAX_DEPTH = 5

with torch.no_grad():
    device = "cpu"
    torch.device("cpu")

    environ["layers"] = "600 350 200 180"
    environ["dropouts"] = "0.05 " * 4

    input_size = 846 #1056
    output_size = 1

    layers_sizes = list(map(int, environ.get("layers", "600 350 200 180").split()))
    drops = list(map(float, environ.get("dropouts", "0.05 0.05 0.05 0.05 0.05").split()))

    model = Model_Recursive_LSTM_v2(
        input_size=input_size,
        comp_embed_layer_sizes=layers_sizes,
        drops=drops,
        loops_tensor_size=8,
        device="cpu",
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    model.to(device)
    model.eval()

    with torch.no_grad():
        try:
            while True:
                
                
                prog_json = input()
                no_sched_json = input()
                sched_json = input()

                program_json = json.loads(prog_json)
                sched_json = json.loads(sched_json)
                no_sched_json = json.loads(no_sched_json)
                (
                prog_tree,
                comps_repr_templates_list,
                loops_repr_templates_list,
                comps_placeholders_indices_dict,
                loops_placeholders_indices_dict
                )  = get_representation_template(program_json, no_sched_json)
                comps_tensor, loops_tensor, comps_expr_repr = get_schedule_representation(
                    program_json,
                    sched_json,
                    comps_repr_templates_list,
                    loops_repr_templates_list,
                    comps_placeholders_indices_dict,
                    loops_placeholders_indices_dict
                )
                
                x = comps_tensor
                batch_size, num_comps, __dict__ = x.shape
                
                x = x.view(batch_size * num_comps, -1)
                
                (first_part, vectors, third_part) = seperate_vector(
                        x, num_transformations=4, pad=False
                    )
                
                first_part = first_part.view(batch_size, num_comps, -1)
                
                third_part = third_part.view(batch_size, num_comps, -1)
                
                tree_tensor = (prog_tree, first_part, vectors, third_part, loops_tensor, comps_expr_repr)

                speedup = model.forward(tree_tensor)
                print(float(speedup.item()))
                
        except EOFError:
            exit()