import streamlit as st
import streamlit.components.v1 as components
import torch
from model import GVAE 
from model import elemList, color_mapping
import json  # Import JSON to handle data conversion
import numpy as np
from codeExplanationPage import codeExplanation

def convert_np_to_python(data):
    if isinstance(data, np.ndarray):
        # Convert numpy array to list of Python types
        return data.tolist()
    elif isinstance(data, np.generic):
        # Convert numpy scalar to Python scalar
        return data.item()
    elif isinstance(data, dict):
        # Recursively apply conversion for dictionary items
        return {key: convert_np_to_python(val) for key, val in data.items()}
    elif isinstance(data, list):
        # Recursively apply conversion for list items
        return [convert_np_to_python(val) for val in data]
    else:
        # Return the data as is if it's already a Python type
        return data



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GVAE().to(device)

checkpoint = torch.load('/Users/avaneeshparasnis/Desktop/TPSP Website/checkpoint.pth.tar', map_location=device)

# Assuming the model's state dictionary is stored under 'model_state_dict'
model_state_dict = checkpoint['model_state_dict']

model.load_state_dict(model_state_dict)

def visualize_crystal(r, elem_list = elemList, color_mapping = color_mapping):
    cart_coords = r.x.cpu().detach().numpy()  # Convert to numpy and detach from GPU/CPU
    edge_index = r.edge_index.t().cpu().detach().numpy()  # Convert edge indices similarly

    atoms_data = [
        {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "color": color_mapping.get(elem_list[int(atom_type)], 'gray')
        }
        for atom_type, x, y, z in cart_coords
    ]

    bonds_data = [
        {
            "start_x": cart_coords[start][1],
            "start_y": cart_coords[start][2],
            "start_z": cart_coords[start][3],
            "end_x": cart_coords[end][1],
            "end_y": cart_coords[end][2],
            "end_z": cart_coords[end][3]
        }
        for start, end in edge_index
    ]

    # Convert data to Python types for JSON serialization
    atoms_data = convert_np_to_python(atoms_data)
    bonds_data = convert_np_to_python(bonds_data)

    # JSON conversion
    atoms_json = json.dumps(atoms_data)
    bonds_json = json.dumps(bonds_data)

    if 'unique_key' not in st.session_state:
        st.session_state['unique_key'] = 0
    st.session_state['unique_key'] += 1
    unique_key = st.session_state['unique_key']

    # HTML and JavaScript integration
    html_template = f"""
    <html>
    <head>
        <script src="https://3dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    </head>
    <body>
        <div style="height: 400px; width: 800px; position: relative;" id="molviewer{unique_key}"></div>
        <script>
            let viewer = new $3Dmol.createViewer('molviewer{unique_key}', {{
                backgroundColor: 'white'
            }});

            let atoms = {atoms_json};
            let bonds = {bonds_json};

            atoms.forEach(atom => {{
                viewer.addSphere({{
                    center: {{x: atom.x, y: atom.y, z: atom.z}},
                    radius: 0.5,
                    color: atom.color
                }});
            }});

            bonds.forEach(bond => {{
                viewer.addCylinder({{
                    start: {{x: bond.start_x, y: bond.start_y, z: bond.start_z}},
                    end: {{x: bond.end_x, y: bond.end_y, z: bond.end_z}},
                    radius: 0.1,
                    color: 'gray'
                }});
            }});

            viewer.zoomTo();
            viewer.render();
        </script>
    </body>
    </html>
    """

    # Display using Streamlit
    components.html(html_template, height=500)

    for atom in cart_coords:
        atom_type = atom[0]  # Assuming atom type is stored under 'type'
        x = atom[1]
        y = atom[2]
        z = atom[3]
        st.text(f"Atom: {elemList[atom_type.astype(np.int64)]}, x: {x}, y: {y}, z: {z}")



# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------


# Set page config to customize the title and the icon
st.set_page_config(page_title="Avaneesh's TPSP Project", page_icon=":test_tube:")

# Add a sidebar for navigation between different pages
st.sidebar.title("Navigation :compass:")
page = st.sidebar.radio(" ", ["Model Showcase 🤖", "Code Explanation 🧠", "Model Comparison 🔍", "Survey 📖"])

if page == "Model Showcase 🤖":
    st.title("Model Showcase 🤖")
    st.header("Crystal Graph Variation Auto-Encoder (CGVAE)")
    shear_modulus = st.number_input('Enter the shear modulus value:', min_value=0.0, max_value=1000.0, value=10.0, key='shear_modulus_1')
    if st.button('Generate Material Graph'):
        with st.spinner('Generating...'):
            graphs = model.sample_graphs(10)
            visualize_crystal(graphs[0])
            st.success('Generated!')
    st.header("Crystal Graph Generative Adversarial Network (CGGAM)")
    shear_modulus = st.number_input('Enter the shear modulus value:', min_value=0.0, max_value=1000.0, value=10.0, key='shear_modulus_2')
    if st.button('Generate Material Graph', key = 'but_1'):
        with st.spinner('Generating...'):
            graphs = model.sample_graphs(10)
            visualize_crystal(graphs[50])
            st.success('Generated!')
    
elif page == "Code Explanation 🧠":
    st.title("Code Explanation 🧠")
    codeExplanation()


elif page == "Model Comparison 🔍":
    st.title("Model Comparison 🔍")
    st.header("CGVAE")
    st.write("The Crystal Graph Variational Auto-Encoder took 4 hours, 56 minutes, and 58 seconds to train on an A100 cloud GPU (a very high-end machine), taking 2 minutes and 50 seconds per epoch on average. It had 1410997 model parameters which is quite large for an AI trained on a personal computer, even while using a cloud GPU. Below is a plot of the training loss by epoch.")
    st.image("CGVAE training.png")
    st.write("As one can see, the model's loss did have a steady trend downward. The notable spikes at various epochs could be due to noisy, outlying data that may not follow the general pattern and cause the model to adjust incorrectly. With the end loss being quite high, there are a number of improvements that can be made:")
    st.markdown("""
    - Constructing a Better Dataset: A cleaner, larger, and more robust dataset would greatly help the training process. The current dataset had ~9000 datapoints that varied greatly. A larger dataset with a smaller degree of variance would help the model learn patterns much more effectively.
    - Hyperparameter Tuning: Changing certain hyperparameters may help the model learn better, however a vast amount of time needs to be spent in this process as the training process for this model is quite demanding.
    - Changing the Model Architecture: A larger, more complex model architecture may allow the model to learn new structures in the data that will help it reduce its loss. However, a larger architecture would mean many more model parameters which would drive trianing time up.
    - Improving the Loss Functions: The loss function may not be optimal for the model to learn how to improve on the data and learn the structure of the materials. More research should go into contructing better loss functions for the models, preferably based on classic materials science principles.
    """)
    st.header("CGGAN")
    st.write("The Crystal Graph Generative Adversarial Network took 1 hours, 3 minutes, and 28 seconds to train on an M1 Macbook Air CPU, taking 38 seconds per epoch on average. It had 50392 model parameters which is lightweight for an AI trained on a personal computer. Below is a plot of the training loss by epoch.")
    st.image("CGGAN.png")
    st.write("Similar to the CGVAE, the model's loss did have a steady trend downward. The extremely jagged nature of this loss tells use that the model was recieving very inconsistent data that it couldn't efficiently adapt to. Note that the model was using a BCE loss function so its loss values may be lower than the CGVAE but in reality, its performance is worse. This is normal however when comparing the sizes and complexities of the two models. This model should use the same improvement suggestions mentioned for the CGVAE.")
    st.header("Comparison")
    st.write("While the CGVAE outperformed the CGGAN in this project, I believe the CGGAN architecture looks more promising for this use case. The CGVAE needs to effectively reconstruct a material from a latent vector. However, without an optimal architecture and strong loss functions, the model will struggle with this reconstruction and creat nonsense materials. The CVGAE struggles to rebuild the lattice coordinates, which is one of the most important features of the material. The CGGAN however was able to learn that similar lattice coordinate to the materials in the dataset makes the materials seems realistic and fool the discriminator, hence learning more about the lattice. With a larger architecture and more research, I believe the CGGAN, or maybe even another diffusion process, can be an effective way to revolutionize the materials design process!")
    

elif page == "Survey 📖":
    st.title("Survey 📖")
    st.header("Thank you so much for viewing my project!!")
    st.write("If possible, please take [this form](https://forms.gle/GRxqLL5MLWo5H93q7) to provide me some insight into what you learned from my project, what you like about my project, and any feedback you may have. Thank you!")
    html_content = """
    <iframe src="https://docs.google.com/forms/d/e/1FAIpQLSfziuOXuCXHJzWnmkV8V_35D82se_4VClE5t6LLv635o3_2LQ/viewform?embedded=true" width="640" height="2124" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>
    """

    # Embed the iframe
    components.html(html_content, height=2124)


