import torch
from torch import nn
import torch.nn.functional as F
from operator import *
from json_to_tensor import MAX_TAGS

# Define the architecture of our cost model
class Model_Recursive_LSTM_v2(nn.Module):
    def __init__(
        self,
        input_size,
        comp_embed_layer_sizes=[600, 350, 200, 180],
        drops=[0.225, 0.225, 0.225, 0.225],
        output_size=1,
        lstm_embedding_size=100,
        expr_embed_size=100,
        loops_tensor_size=20,
        train_device="cpu",
        num_layers=1,
        bidirectional=True,
    ):
        super().__init__()
        self.train_device = train_device
        embedding_size = comp_embed_layer_sizes[-1]
        regression_layer_sizes = [embedding_size] + comp_embed_layer_sizes[-2:]
        concat_layer_sizes = [
            embedding_size * 2 + loops_tensor_size
        ] + comp_embed_layer_sizes[-2:]
        
        comp_embed_layer_sizes = [
            input_size + lstm_embedding_size * (2 if bidirectional else 1) * num_layers + expr_embed_size
        ] + comp_embed_layer_sizes
        self.comp_embedding_layers = nn.ModuleList()
        self.comp_embedding_dropouts = nn.ModuleList()
        self.regression_layers = nn.ModuleList()
        self.regression_dropouts = nn.ModuleList()
        self.concat_layers = nn.ModuleList()
        self.concat_dropouts = nn.ModuleList()
        for i in range(len(comp_embed_layer_sizes) - 1):
            self.comp_embedding_layers.append(
                nn.Linear(
                    comp_embed_layer_sizes[i], comp_embed_layer_sizes[i + 1], bias=True
                )
            )
            nn.init.xavier_uniform_(self.comp_embedding_layers[i].weight)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))
        for i in range(len(regression_layer_sizes) - 1):
            self.regression_layers.append(
                nn.Linear(
                    regression_layer_sizes[i], regression_layer_sizes[i + 1], bias=True
                )
            )
            nn.init.xavier_uniform_(self.regression_layers[i].weight)
            self.regression_dropouts.append(nn.Dropout(drops[i]))
        for i in range(len(concat_layer_sizes) - 1):
            self.concat_layers.append(
                nn.Linear(concat_layer_sizes[i], concat_layer_sizes[i + 1], bias=True)
            )
            nn.init.xavier_uniform_(self.concat_layers[i].weight)
            nn.init.zeros_(self.concat_layers[i].weight)
            self.concat_dropouts.append(nn.Dropout(drops[i]))
        self.predict = nn.Linear(regression_layer_sizes[-1], output_size, bias=True)
        self.encode_vectors = nn.Linear(
            MAX_TAGS,
            MAX_TAGS,
            bias=True,
        )
        nn.init.xavier_uniform_(self.predict.weight)
        self.ELU = nn.ELU()
        self.ReLU = nn.ReLU()
        self.no_comps_tensor = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(1, embedding_size))
        )
        self.no_nodes_tensor = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(1, embedding_size))
        )
        self.root_iterator = nn.Parameter(
                nn.init.xavier_uniform_(torch.zeros(1, loops_tensor_size))
            )
        self.comps_lstm = nn.LSTM(
            comp_embed_layer_sizes[-1], embedding_size, batch_first=True
        )
        self.nodes_lstm = nn.LSTM(
            comp_embed_layer_sizes[-1], embedding_size, batch_first=True
        )
        self.comps_embed = nn.LSTM(
            MAX_TAGS,
            lstm_embedding_size,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )
        self.exprs_embed = nn.LSTM(
            11,
            expr_embed_size,
            batch_first=True,
        )

    def get_hidden_state(self, node, comps_embeddings, loops_tensor):
        nodes_list = []
        if "roots" in node:
            for root in node["roots"]:
                nodes_list.append(self.get_hidden_state(
                    root, comps_embeddings, loops_tensor))
            nodes_tensor = torch.cat(nodes_list, 1)
            lstm_out, (nodes_h_n, nodes_c_n) = self.nodes_lstm(nodes_tensor)
            nodes_h_n = nodes_h_n.permute(1, 0, 2)
            comps_h_n = torch.unsqueeze(self.no_comps_tensor, 0).expand(
                    comps_embeddings.shape[0], -1, -1
                )
            selected_loop_tensor = torch.unsqueeze(self.root_iterator, 0).expand(
                    comps_embeddings.shape[0], -1, -1
                )
        else:
            for n in node["child_list"]:
                nodes_list.append(self.get_hidden_state(
                    n, comps_embeddings, loops_tensor))
            if nodes_list != []:
                nodes_tensor = torch.cat(nodes_list, 1)
                lstm_out, (nodes_h_n, nodes_c_n) = self.nodes_lstm(nodes_tensor)
                nodes_h_n = nodes_h_n.permute(1, 0, 2)
            else:
                nodes_h_n = torch.unsqueeze(self.no_nodes_tensor, 0).expand(
                    comps_embeddings.shape[0], -1, -1
                )
            if node["has_comps"]:
                selected_comps_tensor = torch.index_select(
                    comps_embeddings, 1, node["computations_indices"].to(
                        self.train_device)
                )
                lstm_out, (comps_h_n, comps_c_n) = self.comps_lstm(
                    selected_comps_tensor)
                comps_h_n = comps_h_n.permute(1, 0, 2)
            else:
                comps_h_n = torch.unsqueeze(self.no_comps_tensor, 0).expand(
                    comps_embeddings.shape[0], -1, -1
                )
            selected_loop_tensor = torch.index_select(
                loops_tensor, 1, node["loop_index"].to(self.train_device)
            )
        x = torch.cat((nodes_h_n, comps_h_n, selected_loop_tensor), 2)
        for i in range(len(self.concat_layers)):
            x = self.concat_layers[i](x)
            x = self.concat_dropouts[i](self.ELU(x))
        return x

    def forward(self, tree_tensors):
        tree, comps_tensor_first_part, comps_tensor_vectors, comps_tensor_third_part, loops_tensor, functions_comps_expr_tree, exprs_lengths = tree_tensors
        
        # expressions embedding layer
        x = functions_comps_expr_tree.to(self.train_device)
        batch_size, num_comps, num_expr, expr_len = x.shape
        x = x.view(batch_size * num_comps, num_expr, expr_len)
        x = x.float()
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths=list(exprs_lengths), batch_first=True, enforce_sorted=False)
        _, (expr_embedding, _) = self.exprs_embed(x)
        expr_embedding = expr_embedding.permute(1, 0, 2).reshape(
            batch_size * num_comps, -1
        )
        
        # computation embbedding layer
        # x = comps_tensor.to(self.train_device)
        
        batch_size, num_comps, __dict__ = comps_tensor_first_part.shape
        
        first_part = comps_tensor_first_part.to(self.train_device).view(batch_size * num_comps, -1)
        vectors = comps_tensor_vectors.to(self.train_device) # its shape is (batch_size * num_comps, number of vectors)
        third_part = comps_tensor_third_part.to(self.train_device).view(batch_size * num_comps, -1)
        
        vectors = self.encode_vectors(vectors)
        _, (prog_embedding, _) = self.comps_embed(vectors)

        prog_embedding = prog_embedding.permute(1, 0, 2).reshape(
            batch_size * num_comps, -1
        )
        x = torch.cat(
            (
                first_part,
                prog_embedding,
                third_part,
                expr_embedding,
            ),
            dim=1,
        ).view(batch_size, num_comps, -1)
        
        for i in range(len(self.comp_embedding_layers)):
            x = self.comp_embedding_layers[i](x)
            x = self.comp_embedding_dropouts[i](self.ELU(x))
        comps_embeddings = x
        
        # recursive loop embbeding layer
        prog_embedding = self.get_hidden_state(
            tree, comps_embeddings, loops_tensor.to(self.train_device)
        )
        
        # regression layer
        x = prog_embedding
        for i in range(len(self.regression_layers)):
            x = self.regression_layers[i](x)
            x = self.regression_dropouts[i](self.ELU(x))
        out = self.predict(x)
        return self.ReLU(out[:, 0, 0])