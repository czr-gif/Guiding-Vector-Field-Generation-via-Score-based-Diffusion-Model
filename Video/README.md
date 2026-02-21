# ICRA 2026 Supplementary Video: UAV Trajectory Tracking via Guiding Vector Fields



## 💡 Part 1: Methodology

This video presents the supplementary experimental results for our accepted ICRA 2026 paper. We propose a novel control framework based on Guiding Vector Fields (GVF) for precise UAV trajectory tracking. Compared to traditional methods, our approach ensures that the UAV can smoothly and asymptotically converge to the desired paths even under manual teleoperation interruptions or trajectory switching. 

## 🚁 Part 2: Experiments

* **First Experiment (Concentric Circles):**
  In the first experiment, concentric double-circle trajectories are employed. The UAV initially operates in the inner region, converging to the inner circular path driven by the guiding vector field. Following a manual teleoperation to the outer region, autonomous control is restored, and the UAV subsequently converges to the outer circular path.

* **Second Experiment (Separated Circles):**
  In the second experiment, separated double-circle trajectories are utilized. The UAV first operates in the lower region, converging to the lower circular path. It is then manually teleoperated to the upper region. Upon re-engaging the controller, the UAV successfully converges to the upper circular path.

---

## 📝 Citation

If you find this video or our work helpful, please consider citing our paper:

```bibtex
@inproceedings{...,
  title={...},
  author={...},
  booktitle={2026 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026},
  organization={IEEE}
}