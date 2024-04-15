import torch
import torch.nn as nn
from torch.nn import Linear
import torch_geometric
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn import Set2Set
from torch_geometric.nn import BatchNorm
from tqdm import tqdm
import py3Dmol



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

elemList = ('H',  'Li',  'Be',  'B',  'C',  'N',  'O',  'F',  'Ne',  
            'Na',  'Mg',  'Al',  'Si',  'P',  'S',  'Cl',  'Ar',  'K',  
            'Ca',  'Sc',  'Ti',  'V',  'Cr',  'Mn',  'Fe',  'Co',  'Ni',  
            'Cu',  'Zn',  'Ga',  'Ge',  'As',  'Se',  'Br',  'Kr',  'Rb',  
            'Sr',  'Y',  'Zr',  'Nb',  'Mo',  'Tc',  'Ru',  'Rh',  'Pd',  
            'Ag',  'Cd',  'In',  'Sn',  'Sb',  'Te',  'I',  'Xe',  'Cs',  
            'Ba',  'La',  'Ce',  'Pr',  'Nd',  'Pm',  'Sm',  'Eu',  'Gd',  
            'Tb',  'Dy',  'Ho',  'Er',  'Tm',  'Yb',  'Lu',  'Hf',  'Ta',  
            'W',  'Re',  'Os',  'Ir',  'Pt',  'Au',  'Hg',  'Tl',  'Pb',  
            'Bi',  'Ac',  'Th',  'Pa',  'U',  'Np',  'Pu')
color_mapping = {
      # Alkali metals (Group 1, excluding Hydrogen)
      'Li': 'violet', 'Na': 'violet', 'K': 'violet', 'Rb': 'violet', 'Cs': 'violet', 'Fr': 'violet',
      # Alkaline earth metals (Group 2)
      'Be': 'indigo', 'Mg': 'indigo', 'Ca': 'indigo', 'Sr': 'indigo', 'Ba': 'indigo', 'Ra': 'indigo',
      # Transition metals (Groups 3-12)
      'Sc': 'blue', 'Ti': 'blue', 'V': 'blue', 'Cr': 'blue', 'Mn': 'blue', 'Fe': 'blue',
      'Co': 'blue', 'Ni': 'blue', 'Cu': 'blue', 'Zn': 'blue', 'Y': 'blue',
      'Zr': 'blue', 'Nb': 'blue', 'Mo': 'blue', 'Tc': 'blue', 'Ru': 'blue', 'Rh': 'blue',
      'Pd': 'blue', 'Ag': 'blue', 'Cd': 'blue', 'Hf': 'blue', 'Ta': 'blue', 'W': 'blue',
      'Re': 'blue', 'Os': 'blue', 'Ir': 'blue', 'Pt': 'blue', 'Au': 'blue', 'Hg': 'blue',
      'Rf': 'blue', 'Db': 'blue', 'Sg': 'blue', 'Bh': 'blue', 'Hs': 'blue', 'Mt': 'blue',
      # Post-transition metals
      'Al': 'green', 'Ga': 'green', 'In': 'green', 'Sn': 'green', 'Tl': 'green', 'Pb': 'green', 'Bi': 'green',
      # Metalloids
      'B': 'yellowgreen', 'Si': 'yellowgreen', 'Ge': 'yellowgreen', 'As': 'yellowgreen', 'Sb': 'yellowgreen', 'Te': 'yellowgreen', 'Po': 'yellowgreen',
      # Nonmetals
      'H': 'white', 'C': 'black', 'N': 'blue', 'O': 'red', 'P': 'orange', 'S': 'yellow', 'Se': 'yellow',
      # Halogens (Group 17)
      'F': 'cyan', 'Cl': 'cyan', 'Br': 'cyan', 'I': 'cyan', 'At': 'cyan',
      # Noble gases (Group 18)
      'He': 'magenta', 'Ne': 'magenta', 'Ar': 'magenta', 'Kr': 'magenta', 'Xe': 'magenta', 'Rn': 'magenta',
      # Lanthanides
      'La': 'lightblue', 'Ce': 'lightblue', 'Pr': 'lightblue', 'Nd': 'lightblue', 'Pm': 'lightblue',
      'Sm': 'lightblue', 'Eu': 'lightblue', 'Gd': 'lightblue', 'Tb': 'lightblue', 'Dy': 'lightblue',
      'Ho': 'lightblue', 'Er': 'lightblue', 'Tm': 'lightblue', 'Yb': 'lightblue', 'Lu': 'lightblue',
      # Actinides
      'Ac': 'lightgreen', 'Th': 'lightgreen', 'Pa': 'lightgreen', 'U': 'lightgreen', 'Np': 'lightgreen',
      'Pu': 'lightgreen', 'Am': 'lightgreen', 'Cm': 'lightgreen', 'Bk': 'lightgreen', 'Cf': 'lightgreen',
      'Es': 'lightgreen', 'Fm': 'lightgreen', 'Md': 'lightgreen', 'No': 'lightgreen', 'Lr': 'lightgreen',

  }
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
        self.SM_mean = 50.354785647455095
        self. SM_std = 45.810704083352746

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



    def sample_graphs(self, desired_shear_modulus, num=100):
      print("Sampling materials ... ")

      device = self.device

      desired_shear_modulus = (desired_shear_modulus - self.SM_mean) / self.SM_std
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
