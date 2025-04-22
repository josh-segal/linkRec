1. Load both datasets
2. EDA both - choose winner
3. Train offline policy for 1-3 models
3.1 Top-1
3.2 Top-K
4. Train online policy for 1-3 models
5. Build app
6. Deploy app
7. write report


x. make epsilon-greeby contextual bandit

Web Frontend ←→ API Layer ←→ Model Server
     ↓             ↓            ↓
User Interface  Logging     Model Updates
     ↓             ↓            ↓
Visualizations  Storage    Training Loop

cluster categories for better generalization

