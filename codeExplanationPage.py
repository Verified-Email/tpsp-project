import streamlit as st

def codeExplanation():
    st.sidebar.title("Code Explanation Sections")
    # List of subpages
    subpage = st.sidebar.radio("Select a Topic", ("Dataset", "CGVAE", "CGGAN", "CGVAE Loss Function", "CGVAE Training"))

    # Main Page Content
    if subpage == "Dataset":
        st.header("How I Built My Dataset")
        st.write("My dataset was built using the [Materials Project](https://next-gen.materialsproject.org) database which provides open web-based access to computed information on known and predicted materials. I used the Materials Project API to scrape my desired information from their databases:")
        apiCode = '''
        import pandas as pd
        from mp_api.client import MPRester
        from pymatgen.core import Structure
        from pymatgen.analysis.graphs import StructureGraph
        import matgl
        from matgl.ext.pymatgen import Structure2Graph, get_element_list

        API_KEY = 'My API key'

        mpr = MPRester(API_KEY)

        results = mpr.materials.elasticity.search(fields=["material_id", "structure", 'shear_modulus'])
        data = [result.dict() for result in results]  # Convert result objects to dictionaries

        structureList = [Structure.from_dict(item['structure']) for item in data]

        elemList = get_element_list(structureList)
        '''
        st.code(apiCode, language='python')
        st.write("The imports include MPRester (the tool the Materials Project has put out for scraping their databases), Matgl, and Pymatgen(both of which are materials libraries with many useful functions when working with materials data). I put this scraped data in a list where each materials data was structured in a dictionary. structureList and elemList are variables I use later to convert these dictionaries of materials data to graph structures.")
        datasetCode = '''
        from torch_geometric.utils import to_dense_adj

        s = Structure2Graph(elemList, 4)

        extracted_data = []

        # Iterate over each dictionary in your list
        for item in data:
            # Extract values, including from the subdictionary
            row = {
                'Structure': s.get_graph(Structure.from_dict(item['structure'])),
                'Shear Modulus': item['shear_modulus']['vrh'],
                'Bulk Modulus': item['bulk_modulus']['vrh']
            }
            # Append the extracted data to your list
            extracted_data.append(row)

        df = pd.DataFrame(extracted_data)

        trainingDf = pd.DataFrame(columns = ["Graph", "Structure", "Shear Modulus"])

        trainingDf["Structure"] = df["Structure"]
        trainingDf["Shear Modulus"] = df["Shear Modulus"]

        filtered_df = trainingDf[trainingDf['Shear Modulus'] <= 700]
        filtered_df.reset_index(drop=True, inplace=True)

        filtered_df['Graph'] = filtered_df['Structure'].apply(getGraphFromStructure)

        SM_mean = filtered_df["Shear Modulus"].mean()
        SM_std = filtered_df["Shear Modulus"].std()

        filtered_df["Norm_SM"]=(filtered_df["Shear Modulus"]-SM_mean)/SM_std

        for i in range(len(filtered_df["Graph"])):
        filtered_df["Graph"][i].SM = filtered_df["Norm_SM"][i]
        filtered_df["Graph"][i].edge_index = filter_elements(filtered_df["Graph"][i].edge_index)

        dataset = []

        for graph in filtered_df["Graph"]:
        if(graph.x.size()[0]<=15 and graph.x.size()[0]>2 and graph.x.size()[0] == torch.squeeze(to_dense_adj(graph.edge_index)).size()[0]):
            dataset.append(graph)
        '''
        st.code(datasetCode, language='python')
        st.write("The Structure2Graph object is used to convert the materials data from dictionary form to graph form. I put these graphs, along with the shear modulus data, in a pandas dataframe. I removed any materials with shear moduli greater than 700 as the spread of the data was much too high. I used a custom 'getGraphFromStructure' function to clean the graphs obtained earlier into a form suitable for model. I normalized the shear modulus so the model would be able to interpret it better. I removed any materials with less that 2 atoms and more that 15 atoms to reduce the variability of the materials in the dataset.")
        pytorchDatasetCode = '''
        from torch_geometric.data import Dataset

        class CustomDataset(Dataset):
            def __init__(self, data_list):
                super(CustomDataset, self).__init__()
                self.data_list = data_list

            def len(self):
                return len(self.data_list)

            def get(self, idx):
                return self.data_list[idx]

        MaterialsDataset = CustomDataset(dataset)
        '''
        st.code(pytorchDatasetCode, language='python')
        st.write("Here I converted my final list of material I was using to a Pytorch dataset that could be used during the training process.")


    # Subpage Content
    elif subpage == "CGVAE":
        st.header("Crystal Graph Variational Auto-Encoder Code Explanation")
        st.write("Here is my CGVAE class.")
        initCode = '''
        import torch
        import torch.nn as nn
        from torch.nn import Linear
        from torch_geometric.nn.conv import TransformerConv
        from torch_geometric.nn import Set2Set
        from torch_geometric.nn import BatchNorm
        from tqdm import tqdm

        MAX_MAT_SIZE = 15
        NUM_ATOMS = len(elemList)

        class GVAE(nn.Module):
            def __init__(self, feature_size=4):
                super(GVAE, self).__init__()
                self.encoder_embedding_size = 64
                self.latent_embedding_size = 128
                self.num_atom_types = NUM_ATOMS
                self.max_num_atoms = MAX_MAT_SIZE
                self.decoder_hidden_neurons = 512
                self.device = DEVICE

                # Encoder layers
                self.conv1 = TransformerConv(feature_size,
                                            self.encoder_embedding_size,
                                            heads=4,
                                            concat=False,
                                            beta=True)
                self.bn1 = BatchNorm(self.encoder_embedding_size)
                self.conv2 = TransformerConv(self.encoder_embedding_size,
                                            self.encoder_embedding_size,
                                            heads=4,
                                            concat=False,
                                            beta=True)
                self.bn2 = BatchNorm(self.encoder_embedding_size)
                self.conv3 = TransformerConv(self.encoder_embedding_size,
                                            self.encoder_embedding_size,
                                            heads=4,
                                            concat=False,
                                            beta=True)
                self.bn3 = BatchNorm(self.encoder_embedding_size)
                self.conv4 = TransformerConv(self.encoder_embedding_size,
                                            self.encoder_embedding_size,
                                            heads=4,
                                            concat=False,
                                            beta=True)

                # Pooling layers
                self.pooling = Set2Set(self.encoder_embedding_size, processing_steps=4)

                self.intermediate_linear = Linear(self.encoder_embedding_size * 2, 127)

                # Latent transform layers
                self.mu_transform = Linear(self.latent_embedding_size,
                                                    self.latent_embedding_size)
                self.logvar_transform = Linear(self.latent_embedding_size,
                                                    self.latent_embedding_size)

                # Decoder layers
                # --- Shared layers
                self.linear_1 = Linear(self.latent_embedding_size, self.decoder_hidden_neurons)
                self.linear_2 = Linear(self.decoder_hidden_neurons, self.decoder_hidden_neurons)

                # --- Atom decoding (outputs a matrix: (max_num_atoms) * (# atom_types + "none"-type + x-coord + y-coord + z-coord))
                atom_output_dim = self.max_num_atoms*(self.num_atom_types + 1 + 3)
                self.atom_decode = Linear(self.decoder_hidden_neurons, atom_output_dim)

                # --- Edge decoding (outputs a triu tensor: (max_num_atoms*(max_num_atoms-1)/2))
                edge_output_dim = int(((self.max_num_atoms * (self.max_num_atoms - 1)) / 2) * 2)
                self.edge_decode = Linear(self.decoder_hidden_neurons, edge_output_dim)
                '''
        st.code(initCode, language="python")
        st.write("Here I initialize my model with the layers which include Transformer Convolutions Layers, which utilize multi-head attention, Bath Normalization Layers, used to improve bath training, a Set2Set Layer, used to condense graph data to a vector, and regular Linear Dense Layers.")
        encodingCode = '''
        def encode(self, x, edge_index, shear_modulus, batch_index):
        # GNN layers
        x = self.conv1(x, edge_index).relu()
        x = self.bn1(x)
        x = self.conv2(x, edge_index).relu()
        x = self.bn2(x)
        x = self.conv3(x, edge_index).relu()
        x = self.bn3(x)
        x = self.conv4(x, edge_index).relu()

        # Pool to global representation
        x = self.pooling(x, batch_index)

        #Reduce size to add Shear Modulus
        x = self.intermediate_linear(x)

        shear_modulus = shear_modulus.unsqueeze(-1)

        # Concatenate normalized shear modulus to make it 128 elements
        x = torch.cat((x, shear_modulus), dim=1)

        # Latent transform layers
        mu = self.mu_transform(x)
        logvar = self.logvar_transform(x)
        return mu, logvar
        '''
        st.code(encodingCode, language="python")
        st.write("The encoding process uses Transformer Convolution Layers to extract structure information from the node information (x) and the edge indexes (edge_indexes). \n\nThis is a good time to explain how the graph data is structured. For a graph 'g', 'g.x' is an 'n' by 4 tensor where 'n' is the number of atoms in the material. There are 4 properties for each atom, the atom type (atom index in the element list) and the x, y, and z coordinates of the atoms in the material's crystal lattice unit cell. 'g.edge_index' is a 2 by 'm' tensor where 'm' is the number of bonds the material has. The first subtensor in the 2D tensor contains the indices of the start nodes and the second contains the indices of the end nodes. 'g.SM' is the normalized shear modluls of the material. \n\nAfter the Transformer Convolutions, I use the Set2Set layer to condense the graph to a vector which is then set to a size of 127 through a linear layer. Notice that this is 1 less than the desired latent vector size of 128 because I add the normalized shear modulus as the last element of the vector. This is to let the model learn how the shear modulus affects the structure of the graphs. We finally return the mean and variance of the latent vector as done normally in a conventional Variational Auto-Encoder.")
        decodeCode = '''
        def decode_graph(self, graph_z):
            """
            Decodes a latent vector into a continuous graph representation
            consisting of node types and edge types.
            """
            # Pass through shared layers
            z = self.linear_1(graph_z).relu()
            z = self.linear_2(z).relu()
            # Decode atom types
            atom_logits = self.atom_decode(z)
            # Decode edge types
            edge_logits = self.edge_decode(z)

            return atom_logits, edge_logits


        def decode(self, z, batch_index):
            node_logits = []
            triu_logits = []
            # Iterate over molecules in batch
            for graph_id in torch.unique(batch_index):
                # Get latent vector for this graph
                graph_z = z[graph_id]

                # Recover graph from latent vector
                atom_logits, edge_logits = self.decode_graph(graph_z)

                # Store per graph results
                node_logits.append(atom_logits)
                triu_logits.append(edge_logits)

            # Concatenate all outputs of the batch
            node_logits = torch.cat(node_logits)
            triu_logits = torch.cat(triu_logits)
            return triu_logits, node_logits
        '''
        st.code(decodeCode, language="python")
        st.write("The decoder utilizes Linear layers to shape the inputed latent vector into node and edge information that can be restructured into a graph with the same format as the data in the training dataset. node_logits is a 15 by 92 tensor which encapsulates all the data necessary to reconstruct the 'g.x' part of a material. The 15 is the maximum number of atoms a material can have. This choice is kept at this level to reduce the complexity of the model and to encompass the most training data possible. The 92 represents the number of atoms choices available to be in the material (len(elemList)) plus 1 to represent no atom (as materials don't need to have the maximum 15 atoms) and then plus 3 for the x, y, and z coordinates of that atom. trui_logits is all that is needed to reconstruct the 'g.edge_index' of a material. It is the upper right corner of the adgacency matrix of the material graph. We only need the upper right hand corner as each atom pair can only have a maximum of one edge.")
        forwardCode = '''
        def reparameterize(self, mu, logvar):
        if self.training:
            # Get standard deviation
            std = torch.exp(logvar)
            # Returns random numbers from a normal distribution
            eps = torch.randn_like(std)
            # Return sampled values
            return eps.mul(std).add_(mu)
        else:
            return mu

        def forward(self, x, edge_index, shear_modulus, batch_index):
        # Encode the molecule
        mu, logvar = self.encode(x, edge_index, shear_modulus, batch_index)
        # Sample latent vector (per atom)
        z = self.reparameterize(mu, logvar)
        # Decode latent vector into original molecule
        triu_logits, node_logits = self.decode(z, batch_index)
        return triu_logits, node_logits, mu, logvar
        '''
        st.code(forwardCode, language='python')
        st.write("The forward function connects the encoder and decoder and utilizes the reparameterization trick necessary for VAEs. It returns all necessary info for calculating the loss of the training epoch.")
        sampleCode = '''
        def sample_graphs(self, desired_shear_modulus, num=100):
      print("Sampling materials ... ")

      device = self.device

      desired_shear_modulus = (desired_shear_modulus - SM_mean) / SM_std
      desired_shear_modulus = torch.tensor([desired_shear_modulus], dtype=torch.float32, device=device)

      mats = []

      # Sample materials and check if they are valid
      for _ in tqdm(range(num)):
          # Sample latent space
          z = torch.randn(1, self.latent_embedding_size - 1, device=device)

          # Adjust dimensions of desired_shear_modulus
          desired_shear_modulus_unsqueezed = desired_shear_modulus.unsqueeze(1)

          # Concatenate normalized shear modulus to make it 128 elements
          z = torch.cat((z, desired_shear_modulus_unsqueezed), dim=1)

          # Get model output (this could also be batched)
          dummy_batch_index = torch.tensor([0], dtype=torch.int32, device=device)
          t, n = self.decode(z, dummy_batch_index)

          node_matrix_shape = (MAX_MAT_SIZE, (NUM_ATOMS + 1 + 3))
          node_preds_matrix = n.view(node_matrix_shape)
          node_preds = torch.argmax(node_preds_matrix[:, :-3], dim=1)
          node_coords = node_preds_matrix[:, -3:]

          node_preds_reshaped = node_preds.to(node_coords.dtype).unsqueeze(1)
          node_features = torch.cat((node_preds_reshaped, node_coords), dim=1)

          edge_matrix_shape = (int((MAX_MAT_SIZE * (MAX_MAT_SIZE - 1)) / 2), 2)
          triu_preds_matrix = t.view(edge_matrix_shape)
          triu_preds = torch.argmax(triu_preds_matrix, dim=1)

          edges = torch.tensor([[0, 0]], device=device)

          index = 0
          for i in range(15):
              for j in range(i+1, 15):
                  if triu_preds[index] == 1 and node_preds[i] != 88 and node_preds[j] != 88:
                      edge = torch.tensor([[i, j]], device=device)
                      edges = torch.cat((edges, edge), dim=0)
                  index += 1

          edges = edges[1:].t()

          index_to_remove = torch.where(node_preds == 88)[0]
          for ind in reversed(index_to_remove):
              mask = torch.arange(node_features.size(0), device=device) != ind

              # Apply the mask
              node_features = node_features[mask]
              edges = edges[(edges[:, 0] != ind) & (edges[:, 1] != ind)]

              # Decrement indices of nodes after the removed node
              edges[edges >= ind] -= 1

          gen_graph = torch_geometric.data.Data(x=node_features, edge_index=edges)
          mats.append(gen_graph)

      return mats
        '''
        st.code(sampleCode, language='python')
        st.write("This final function is used to create novel materials after the model is done training. It creates a random 127 element latent vector and then appends the desired shear modulus value after being normalized so that the model constructs materials that hopefully have a structure that permits the desired material property. The function then reshapes the logits returned by the decoding process into a functional material graph representation. The function does this 100 times and stores these generated graphs into a list that gets returned. Researchers can then use these sampled materials in the materials design process!")

    elif subpage == "CGGAN":
        st.header("Crystal Graph Generative Adversarial Network Code Explanation")
        st.write("Here is the code for the CGGAN. This model is much more lightweight due to time constraints.")
        genCode = '''
        class GraphGenerator(nn.Module):
    def __init__(self, feature_size=4):
        super(GraphGenerator, self).__init__()
        self.encoder_embedding_size = 64
        self.latent_embedding_size = 128
        self.num_atom_types = NUM_ATOMS
        self.max_num_atoms = MAX_MAT_SIZE
        self.decoder_hidden_neurons = 512

        # Latent space dimension includes shear modulus
        self.latent_dim = self.latent_embedding_size + 1

        # Generator layers
        self.linear_1 = Linear(self.latent_dim, self.decoder_hidden_neurons)
        self.linear_2 = Linear(self.decoder_hidden_neurons, self.decoder_hidden_neurons)

        # Atom decoding
        atom_output_dim = self.max_num_atoms * (self.num_atom_types + 1 + 3)
        self.atom_decode = Linear(self.decoder_hidden_neurons, atom_output_dim)

        # Edge decoding
        edge_output_dim = int(((self.max_num_atoms * (self.max_num_atoms - 1)) / 2) * 2)
        self.edge_decode = Linear(self.decoder_hidden_neurons, edge_output_dim)

    def forward(self, z):
        z = z.relu()
        z = self.linear_1(z).relu()
        z = self.linear_2(z).relu()

        atom_logits = self.atom_decode(z)
        edge_logits = self.edge_decode(z)

        return atom_logits, edge_logits
        '''
        st.code(genCode, language='python')
        st.write("This model uses simple Linear layers to convert a latent vector to the proper logit sizes for the nodes and edges as shown in the CGVAE explanation.")
        discCode = '''
        class GraphDiscriminator(nn.Module):
    def __init__(self, feature_size=4):
        super(GraphDiscriminator, self).__init__()
        self.encoder_embedding_size = 64

        # Encoder layers
        self.conv1 = TransformerConv(feature_size,
                                     self.encoder_embedding_size,
                                     heads=4, concat=False, beta=True)
        self.bn1 = BatchNorm(self.encoder_embedding_size)

        # Pooling
        self.pooling = Set2Set(self.encoder_embedding_size, processing_steps=4)
        
        # Classification layer
        self.classifier = Linear(self.encoder_embedding_size * 2, 1)

    def forward(self, x, edge_index, batch_index):
        x = self.conv1(x, edge_index).relu()
        x = self.bn1(x)
        x = self.pooling(x, batch_index)
        
        return torch.sigmoid(self.classifier(x))
        '''
        st.code(discCode, language='python')
        st.write("The discriminator uses a Transformer Convolution Layer, a Pooling Layer, and a Linear layer to convert a graph into a single element to dircriminate the validity of the input graph")
        trainGANcode = '''
        def train_gan(generator, discriminator, data_loader, device=DEVICE):
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(50):  # Example epoch count
        for data in data_loader:
            # Train Discriminator
            optimizer_D.zero_grad()
            real_data = data.to(device)
            real_output = discriminator(real_data.x, real_data.edge_index, real_data.batch)
            real_label = torch.ones(real_output.shape[0], 1, device=device)
            loss_D_real = criterion(real_output, real_label)

            z = torch.randn(real_data.num_graphs, generator.latent_dim, device=device)
            generated_atoms, generated_edges = generator(z)
            fake_output = discriminator(generated_atoms, generated_edges, real_data.batch)
            fake_label = torch.zeros(fake_output.shape[0], 1, device=device)
            loss_D_fake = criterion(fake_output, fake_label)

            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(real_data.num_graphs, generator.latent_dim, device=device)
            generated_atoms, generated_edges = generator(z)
            fake_output = discriminator(generated_atoms, generated_edges, real_data.batch)
            loss_G = criterion(fake_output, real_label)
            loss_G.backward()
            optimizer_G.step()

            print(f"Epoch {epoch}, Loss D: {loss_D.item()}, Loss G: {loss_G.item()}")
        '''
        st.code(trainGANcode,language='python')
        st.write("Because this model is so lightweight, the loss function and training is very simplistic. I used a BCE loss function and the ADAM optimizer.")

    elif subpage == "CGVAE Loss Function":
        st.header("CGVAE Loss Function Explanation")
        st.write("There are numerous helper functions for basic tasks for this part of my code which I won't discuss here. You can examine my original code on GitHub to examine the functionality of these functions.")
        approxCode = '''
        def approximate_recon_loss(node_targets, node_preds, triu_targets, triu_preds):
    atom_targets = node_targets[:,:1]
    coord_targets = node_targets[:,1:]

    # Convert targets to one hot
    onehot_node_targets = to_one_hot(atom_targets, elemList ) #+ ["None"]
    onehot_triu_targets = to_one_hot(triu_targets, ["None", "Edge"])

    # Reshape node predictions
    node_matrix_shape = (MAX_MAT_SIZE, (len(elemList) + 1 + 3))
    node_preds_matrix = node_preds.reshape(node_matrix_shape)

    # Reshape triu predictions
    edge_matrix_shape = (int((MAX_MAT_SIZE * (MAX_MAT_SIZE - 1))/2), 2)
    triu_preds_matrix = triu_preds.reshape(edge_matrix_shape)

    # Apply sum on labels per (node/edge) type and discard "none" types
    node_preds_reduced = torch.sum(node_preds_matrix[:, :88], 0)
    node_targets_reduced = torch.sum(onehot_node_targets, 0)
    triu_preds_reduced = torch.sum(triu_preds_matrix[:, 1:], 0)
    triu_targets_reduced = torch.sum(onehot_triu_targets[:, 1:], 0)

    # Calculate node-sum loss and edge-sum loss
    node_loss = torch.sum(squared_difference(node_preds_reduced, node_targets_reduced.float()))
    edge_loss = torch.sum(squared_difference(triu_preds_reduced, triu_targets_reduced.float()))

    # Calculate coordinate loss
    coord_target_matrix = torch.zeros(88, 3).to(device)

    for node in node_targets:
      coord_target_matrix[node[0].int()][0] += node[1]
      coord_target_matrix[node[0].int()][1] += node[2]
      coord_target_matrix[node[0].int()][2] += node[3]

    atom_preds = torch.argmax(node_preds_matrix[:, :-3], dim=1)
    atom_preds = atom_preds.unsqueeze(1)
    node_coords = node_preds_matrix[:, -3:]
    node_preds_coord_matrix = torch.cat((atom_preds, node_coords), dim=1)

    coord_pred_matrix = torch.zeros(88, 3).to(device)

    for node in node_preds_coord_matrix:
      if node[0].int() != 88:
        coord_pred_matrix[node[0].int()][0] += node[1]
        coord_pred_matrix[node[0].int()][1] += node[2]
        coord_pred_matrix[node[0].int()][2] += node[3]

    coord_loss = torch.sum(squared_difference(coord_pred_matrix.float(), coord_target_matrix.float()))


    # Calculate node-edge-sum loss
    # Forces the model to properly arrange the matrices
    node_edge_loss = calculate_node_edge_pair_loss(onehot_node_targets,
                                      onehot_triu_targets,
                                      node_preds_matrix[:, :89],
                                      triu_preds_matrix)

    approx_loss =   node_loss + coord_loss + edge_loss + node_edge_loss
    return approx_loss
        '''
        st.code(approxCode, language='python')
        st.write("This function is called in the main loss function. It is used to see how close the generated material was to its corresponding input material. It is made up of four independent losses added together at the end: the Node Loss, the Coordinate Loss, the Edge Loss, and the Node-Edge Loss. I will explain these losses at a high level as many basic tensor calculations are needed during these step which would make emplaining the entirety of the code time not particularily useful.\n\nNode Loss:\nThe Node Loss is created unsing 88x1 tensors for the input graph and the generated graph, detailing the number of specific elements are in the material. For example, if carbon was at index 3 in the element list and there were 6 carbon atoms in the material, then the third index of the tensor would be 6. The Node Loss is formed by taking the squared difference of the two tensors (input and generated) and then adding all elements of this summation matrix together. This is supposed to push the model to reconstruct a material with the same elements the input material has.\n\nCoordinate Loss:\nThe Coordinate Loss is created by creating two 88x3 tensors called 'coord_target_matrix' and 'coord_pred_matrix'. For each of a specific element in the input graph of index 'i' in the element list, 'coord_target_matrix[i][0]' is the sum of the x-coordinates of all atoms of that specifc element type in the material. The same is done for the y and z coordinates of the graph in 'coord_target_matrix[i][1]' and 'coord_target_matrix[i][2]' respectively. The same exact process occurs for the generated material in 'coord_pred_matrix'. The Coordinate Loss is the sum of all the elements of the tensor obtained by taking the squared difference of 'coord_pred_matrix' and 'coord_target_matrix'.\n\nEdge Loss:\nThe Edge Loss is calculated by taking the squared difference of the number of edges in the generated graph and the input graph.\n\nThe Node-Edge Loss is calculated by a call to the 'calculate_node_edge_pair_loss' function explained below.")
        nodeEdgeCode = '''
        def calculate_node_edge_pair_loss(node_tar, edge_tar, node_pred, edge_pred):
    """
    Calculates a loss based on the sum of node-edge pairs.
    node_tar:  [nodes, supported atoms]
    node_pred: [max nodes, supported atoms + 1]
    edge_tar:  [triu values for target nodes, supported edges]
    edge_pred: [triu values for predicted nodes, supported edges]

    """
    # Recover full 3d adjacency matrix for edge predictions
    edge_pred_mat = triu_to_dense(edge_pred[:,1].float(), node_pred.shape[0]) # [num nodes, num nodes]

    # Recover full 3d adjacency matrix for edge targets
    edge_tar_mat = triu_to_dense(edge_tar[:,1].float(), node_tar.shape[0]) # [num nodes, num nodes]

    # --- The two output matrices tell us how many edges are connected with each of the atom types
    # Multiply each of the edge types with the atom types for the predictions
    node_edge_preds = torch.empty((MAX_MAT_SIZE, len(elemList)), dtype=torch.float, device=device)
    node_edge_preds = torch.matmul(edge_pred_mat, node_pred[:, :88])

    # Multiply each of the edge types with the atom types for the targets
    node_edge_tar = torch.empty((node_tar.shape[0], len(elemList)), dtype=torch.float, device=device)
    node_edge_tar = torch.matmul(edge_tar_mat, node_tar.float().squeeze())

    # Reduce to matrix with [num atom types, num edge types]
    node_edge_pred_matrix = torch.sum(node_edge_preds, dim=0)
    node_edge_tar_matrix = torch.sum(node_edge_tar, dim=0)

    node_edge_loss = torch.mean(sum(squared_difference(node_edge_pred_matrix, node_edge_tar_matrix.float())))

    return node_edge_loss
        '''
        st.code(nodeEdgeCode, language='python')
        st.write("The Node-Edge Loss is sum of all elements of the squared difference of the 'node_edge_pred_matrix' and the 'node_edge_tar_matrix'. These matrices detail the number of edges a specific element has in material graph for the generated graph and the input graph respectively. This loss is used for the model to understand the structure of the bonds/edges.")
        gvaeLossCode = '''
        def gvae_loss(triu_logits, node_logits, edge_index, node_types, \
              mu, logvar, batch_index, kl_beta):
    """
    Calculates the loss for the graph variational autoencoder,
    consiting of a node loss, an edge loss and the KL divergence.
    """
    # Convert target edge index to dense adjacency matrix
    batch_edge_targets = torch.squeeze(to_dense_adj(edge_index))

    # For this model we always have the same (fixed) output dimension
    graph_size = MAX_MAT_SIZE*(len(elemList) + 1+3)
    graph_triu_size = int((MAX_MAT_SIZE * (MAX_MAT_SIZE - 1)) / 2) * 2

    # Reconstruction loss per graph
    batch_recon_loss = []
    triu_indices_counter = 0
    graph_size_counter = 0

    # Loop over graphs in this batch
    for graph_id in torch.unique(batch_index):
            # Get upper triangular targets for this graph from the whole batch
            triu_targets, node_targets = slice_graph_targets(graph_id,
                                                            batch_edge_targets,
                                                            node_types,
                                                            batch_index)

            # Get upper triangular predictions for this graph from the whole batch
            triu_preds, node_preds = slice_graph_predictions(triu_logits,
                                                            node_logits,
                                                            graph_triu_size,
                                                            triu_indices_counter,
                                                            graph_size,
                                                            graph_size_counter)

            # Update counter to the index of the next (upper-triu) graph
            triu_indices_counter = triu_indices_counter + graph_triu_size
            graph_size_counter = graph_size_counter + graph_size

            # Calculate losses
            recon_loss = approximate_recon_loss(node_targets,
                                                node_preds,
                                                triu_targets,
                                                triu_preds)
            batch_recon_loss.append(recon_loss)

    # Take average of all losses
    num_graphs = torch.unique(batch_index).shape[0]
    batch_recon_loss = torch.true_divide(sum(batch_recon_loss),  num_graphs)

    # KL Divergence
    kl_divergence = kl_loss(mu, logvar)

    return batch_recon_loss + kl_beta * kl_divergence, kl_divergence
        '''
        st.code(gvaeLossCode, language='python')
        st.write("This code finds the 'kl_divergence', a classic part of a Variational Auto-Encoder, using some helper function. It also calculate the Approximate Reconstruction Loss (the sum of the four losses explained above) by formating the data from the batch process.")


    elif subpage == "CGVAE Training":
        st.header("How I Trained the CGVAE Model")
        trainCode = '''
        import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm
import numpy as np


# Load data
train_dataset = MaterialsDataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load model
model = GVAE()
model = model.to(device)
print("Model parameters: ", count_parameters(model))

# Define loss and optimizer
loss_fn = gvae_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
kl_beta = 0.5

# Train function
def run_one_epoch(data_loader, type, epoch, kl_beta):
    # Store per batch loss and accuracy
    all_losses = []
    all_kldivs = []

    # Iterate over data loader
    for _, batch in enumerate(tqdm(data_loader)):
            # Use GPU
            batch.to(device)
            # Reset gradients
            optimizer.zero_grad()
            # Call model
            triu_logits, node_logits, mu, logvar = model(batch.x.float(),
                                                        batch.edge_index,
                                                        batch.SM.float(),
                                                        batch.batch)
            # Calculate loss and backpropagate
            loss, kl_div = loss_fn(triu_logits, node_logits,
                                   batch.edge_index,
                                   batch.x.float(), mu, logvar,
                                   batch.batch, kl_beta)
            if type == "Train":
                loss.backward()
                optimizer.step()
            # Store loss and metrics
            all_losses.append(loss.detach().cpu().numpy())
            #all_accs.append(acc)
            all_kldivs.append(kl_div.detach().cpu().numpy())

    print(f"{type} epoch {epoch} loss: ", np.array(all_losses).mean())
    mlflow.log_metric(key=f"{type} Epoch Loss", value=float(np.array(all_losses).mean()), step=epoch)
    mlflow.log_metric(key=f"{type} KL Divergence", value=float(np.array(all_kldivs).mean()), step=epoch)
    mlflow.pytorch.log_model(model, "model")

# Run training
with mlflow.start_run() as run:
    for epoch in range(101):
        checkpoint = {
          "epoch": epoch,
          "model_state_dict": model.state_dict(),
          "optimizer_state_dict": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        model.train()
        run_one_epoch(train_loader, type="Train", epoch=epoch, kl_beta=kl_beta)

    mlflow.pytorch.log_model(model, "model")
        '''
        st.code(trainCode, language='python')
        st.write("This code sets up a basic epoch training loop. I save the model's parameters each epoch and log the loss in Mlflow.")
