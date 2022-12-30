import torch
from torch import nn


def seperate_vector(
    X: torch.Tensor, num_matrices: int = 4, pad: bool = True, pad_amount: int = 5
) -> torch.Tensor:
    batch_size, _ = X.shape
    tags_num =  (6*num_matrices+3)
    first_part = X[:, :tags_num]
    second_part = X[:, tags_num : tags_num + 36 * num_matrices]
    third_part = X[:, tags_num + 36 * num_matrices :]
    vectors = []
    for i in range(num_matrices):
        vector = second_part[:, 36 * i : 36 * (i + 1)].reshape(batch_size, 1, -1)
        vectors.append(vector)

    if pad:
        for i in range(pad_amount):
            vector = torch.zeros_like(vector)
            vectors.append(vector)
    return (first_part, vectors[0], torch.cat(vectors[1:], dim=1), third_part)


class Model_Recursive_LSTM_v2(nn.Module):
    def __init__(
        self,
        input_size=776,
        comp_embed_layer_sizes=[600, 350, 200, 180],
        drops=[0.05, 0.05, 0.05, 0.05],
        output_size=1,
        lstm_embedding_size=100,
        transformation_matrix_dimension=6,
        loops_tensor_size=20,
        train_device="cpu",
        num_layers=1,
        bidirectional=True,
        embedding_generator=None,
    ):
        super().__init__()
        self.train_device = train_device
        embedding_size = comp_embed_layer_sizes[-1]
        regression_layer_sizes = [embedding_size] + comp_embed_layer_sizes[-2:]
        concat_layer_sizes = [
            embedding_size * 2 + loops_tensor_size
        ] + comp_embed_layer_sizes[-2:]
        comp_embed_layer_sizes = [
            input_size
            + lstm_embedding_size * (2 if bidirectional else 1) * num_layers
            #   -
            #   transformation_matrix_dimension**2
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
            # nn.init.zeros_(self.comp_embedding_layers[i].weight)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))
        for i in range(len(regression_layer_sizes) - 1):
            self.regression_layers.append(
                nn.Linear(
                    regression_layer_sizes[i], regression_layer_sizes[i + 1], bias=True
                )
            )
            nn.init.xavier_uniform_(self.regression_layers[i].weight)
            # nn.init.zeros_(self.regression_layers[i].weight)
            self.regression_dropouts.append(nn.Dropout(drops[i]))
        for i in range(len(concat_layer_sizes) - 1):
            self.concat_layers.append(
                nn.Linear(concat_layer_sizes[i], concat_layer_sizes[i + 1], bias=True)
            )
            nn.init.xavier_uniform_(self.concat_layers[i].weight)
            nn.init.zeros_(self.concat_layers[i].weight)
            self.concat_dropouts.append(nn.Dropout(drops[i]))
        self.predict = nn.Linear(regression_layer_sizes[-1], output_size, bias=True)
        self.encode_vectors = nn.Linear(transformation_matrix_dimension**2, transformation_matrix_dimension**2, bias=True)
        nn.init.xavier_uniform_(self.predict.weight)
        # nn.init.zeros_(self.predict.weight)
        self.ELU = nn.ELU()
        self.ReLU = nn.ReLU()
        self.no_comps_tensor = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(1, embedding_size))
        )
        self.no_nodes_tensor = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(1, embedding_size))
        )
        self.comps_lstm = nn.LSTM(
            comp_embed_layer_sizes[-1], embedding_size, batch_first=True
        )
        self.nodes_lstm = nn.LSTM(
            comp_embed_layer_sizes[-1], embedding_size, batch_first=True
        )
        self.comps_embed = nn.LSTM(
            transformation_matrix_dimension**2,
            lstm_embedding_size,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )
        # self._reinitialize()

        if embedding_generator:
            self.embedding_generator = embedding_generator
        else:
            self.embedding_generator = lambda x: x



    def get_hidden_state(self, node, comps_embeddings, loops_tensor):
        nodes_list = []
        for n in node["child_list"]:
            nodes_list.append(self.get_hidden_state(n, comps_embeddings, loops_tensor))
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
                comps_embeddings, 1, torch.tensor(node["computations_indices"]).to(self.train_device)
            )
            lstm_out, (comps_h_n, comps_c_n) = self.comps_lstm(selected_comps_tensor)
            comps_h_n = comps_h_n.permute(1, 0, 2)
        else:
            comps_h_n = torch.unsqueeze(self.no_comps_tensor, 0).expand(
                comps_embeddings.shape[0], -1, -1
            )
        selected_loop_tensor = torch.index_select(
            loops_tensor, 1, torch.tensor(node["loop_index"]).to(self.train_device)
        )
        x = torch.cat((nodes_h_n, comps_h_n, selected_loop_tensor), 2)
        for i in range(len(self.concat_layers)):
            x = self.concat_layers[i](x)
            x = self.concat_dropouts[i](self.ELU(x))
        return x

    def forward(self, tree_tensors, num_matrices=6):
        tree, comps_tensor, loops_tensor = tree_tensors
        # computation embbedding layer
        x = comps_tensor.to(self.train_device)

        batch_size, num_comps, input_size = x.shape
        x = x.view(batch_size * num_comps, -1)
        (first_part, final_matrix, vectors, third_part) = seperate_vector(
            x, num_matrices=num_matrices, pad=False
        )
        # print(first_part.shape, final_matrix.shape, vectors.shape, third_part.shape)
        # mask = (vectors == 0).sum(dim=1).sum(dim=1) == 288
        # vectors = vectors[~mask,:,:]
        # final_matrix = final_matrix[~mask,:,:]
        # batch_size, num_comps, _ = final_matrix.shape
        vectors =  self.encode_vectors(vectors)
        # self.ELU(vectors)
        # print(vectors.shape)
        lstm_out, (prog_embedding, comps_c_n) = self.comps_embed(vectors)
        # print(vectors.shape)

        prog_embedding = prog_embedding.permute(1, 0, 2).reshape(
            batch_size * num_comps, -1
        )
        # self.ELU(prog_embedding)
        # print(prog_embedding.shape)
        # prog_embedding = self.embedding_generator(vectors.cuda())
        # prog_embedding = prog_embedding.reshape(batch_size*num_comps, -1)
        x = torch.cat(
            (
                first_part,
                final_matrix.reshape(batch_size * num_comps, -1),
                # vectors.reshape(batch_size*num_comps, -1),
                prog_embedding,
                third_part,
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
        return self.ELU(out[:, 0, 0])

