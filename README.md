# Guiding Vector Field Generation via Score-based Diffusion Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ICRA 2026](https://img.shields.io/badge/ICRA_2026-Accepted-blue.svg)](https://2026.ieee-icra.org/)

This is the official PyTorch implementation for the **ICRA 2026** paper:  
**Guiding Vector Field Generation via Score-based Diffusion Model**

**Authors:** Zirui Chen, Shiliang Guo, Shiyu Zhao  
**Affiliation:** Westlake University & Zhejiang University

---

## 📢 News
* **[2026-02-21]** 📄 The camera-ready version has been submitted! Full paper is available at https://www.researchgate.net/publication/401018693_Guiding_Vector_Field_Generation_via_Score-based_Diffusion_Model
* **[2026-02-01]** 🎉 Our paper has been accepted by **ICRA 2026**!
* **[2026-02-01]** Code released.

---

## 📁 Project Structure

The repository is organized as follows:

```text
.
├── configs/                # Configuration files (default.yaml)
├── core/                   # Core modules (dynamics, paths, diffusion schedules, simulators)
├── experiment_data/        # 📊 Raw flight logs (.txt) and trajectory plots (.pdf) from real UAV experiments
├── models/                 # Neural network architectures (Score and Tangent networks)
├── saved_figure/           # Visualizations and loss curves for various shapes/tasks
├── saved_model/            # Pre-trained PyTorch weights (.pth) 
├── scripts/                # Executable scripts for training and simulation
│   ├── train_score.py      # Training script for the score network
│   ├── train_tangent.py    # Training script for the tangent network
│   ├── path_following.py   # Simulation script for UAV path following
│   └── ...                 # Ablation studies and visualization scripts
├── trans_model/            # Scripts for exporting models to ONNX/RKNN for UAV hardware deployment
├── utils/                  # Utility functions (math, IO, plotting)
├── Video/                  # 🎬 Supplementary video demonstrations for the paper
├── environment.yml         # Conda environment dependencies
└── README.md
```

(Note: Folders like saved_model and saved_figure contain sub-folders corresponding to specific tasks such as circle, doublecircles, experiment1, experiment2, etc.)

🚀 Quick Start
1. Environment Setup
We recommend using Conda to manage the environment. You can recreate the environment using the provided environment.yml file:
```
conda env create -f environment.yml
conda activate diffusion  # Replace 'diffusion' with your environment name if changed
```

2. Training the Models
To train the Score network and the Tangent network from scratch, run the scripts as modules from the root directory:

```
# Train the score-based diffusion model
python -m scripts.train_score

# Train the tangent vector field network
python -m scripts.train_tangent
```

3. Running Simulations
To reproduce the path-following simulations (e.g., concentric circles, separated circles, polygons), use the path_following.py script:
```
python -m scripts.path_following
```
💡 Note on switching tasks: To simulate different trajectories, open scripts/path_following.py and modify the taskname variable inside the script (e.g., change it to 'experiment1', 'circle', 'Square', etc.) before running. The script will automatically load the corresponding pre-trained weights from the saved_model folder.

4. Real-world Experiments & Video
    Experimental Data: The raw telemetry data (.txt files) and the plotted results (.pdf) from our real-world UAV flight tests are available in the experiment_data/ folder.

    Demonstration: Please check the Video/ folder for the full supplementary video showing the UAV executing trajectory tracking and switching in reality.


## 📝 Citation

If you find this work useful for your research, please cite our paper.  
*(Note: Full proceedings details including page numbers and DOI will be updated once available.)*

```bibtex
@inproceedings{chen2026guiding,
  title={Guiding Vector Field Generation via Score-based Diffusion Model},
  author={Chen, Zirui and Guo, Shiliang and Zhao, Shiyu},
  booktitle={2026 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026},
  note={Accepted}
}
