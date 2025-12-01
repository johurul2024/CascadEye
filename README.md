# CascadEye

CSC792 – Machine Learning System Design  
Department of Electrical Engineering and Computer Science  
South Dakota State University, USA  

**Team:** Syed Abdullah-Al Nahid, Mohammad Johurul Islam, Yingman Bian  
**Course Instructor:** Prof. Chulwoo Pack

CascadEye is a lightweight desktop ML system for cascading-failure risk analysis in power grids.  
It turns historical outage records into:

- an interaction matrix **B** between transmission lines  
- Monte-Carlo cascade simulations  
- critical-line rankings  
- basic monitoring plots (PDF/CCDF, KS, directional test, etc.)  

The tool is implemented as an offline Tkinter desktop app, with a pretrained LSTM + Multi-Head Attention model.

---

## Project structure

- `src/cascadeye/` – app + pipeline code (Tkinter UI, data pipeline)  
- `models/` – pretrained model checkpoints (`.pth`) and small metrics files  
- `data/` – small demo outage CSVs (original + synthetic variants)  
- `artifacts/` – per-run outputs (logs, B matrices, Top-k tables, stats)  
- `project_reports/` – Jupyter notebook + exported HTML report + exported PDF report
- `figures/` – figures used in the notebook / report  

---

## Requirements & setup (local)

Tested with:

- Python 3.10+  
- `numpy`, `pandas`, `scipy`  
- `matplotlib`  
- `torch`  
- `tkinter`

**Note:** When you download the report, keep `cascadeye.ipynb` or `cascadeye.html` and the `figures/` folder in the **same directory**; otherwise the images will not display correctly.
