# EdenMaze  
### A Multi-Agent Reinforcement Learning System with Dynamic Rule Generation and Full Azure Deployment

---

## Overview

EdenMaze is an end-to-end machine learning system that simulates a dynamic multi-agent trading environment.

In a virtual environment:

- 25 AI agents interact pairwise
- Each agent chooses between **cooperation** or **defection**
- Outcomes are recorded sequentially (action, reward, transaction ID)
- Agents are trained via **Reinforcement Learning**
- A **Generative Rule Model** dynamically adjusts game rules and environment parameters

The system includes:

- Multi-agent RL training framework
- Rule generation module
- Evaluation and metric tracking
- Data persistence and experiment logging
- Web-based dashboard for real-time visualization
- Project landing site
- Azure cloud deployment (ML, storage, monitoring, frontend)

---

## System Architecture
User (Web Dashboard)
↓
Azure Container API (Trigger / Query)
↓
Azure Machine Learning Job
↓
Multi-Agent Simulation (RL + Rule Generator)
↓
Artifacts + Metrics → Azure Data Lake
↓
Monitoring → Azure Monitor / App Insights


### Components

- **ML Core** → Multi-agent simulation & training
- **API Layer** → Triggers jobs, exposes metrics & logs
- **Dashboard Web App** → Visualizes results
- **Landing Site** → Project presentation
- **Azure Infrastructure** → Compute, storage, monitoring

---

## Repository Structure

.
├── docs/ # Architecture and design documentation
├── homepage/ # Landing page (project presentation)
├── web/ # Interactive dashboard
├── ml/ # ML training & simulation system
│ ├── agents/
│ ├── configs/
│ ├── src/
│ ├── data/
│ ├── designer/
│ └── visual/
└── infra/ # Infrastructure as code (optional)


---

## Core Features

### 1. Multi-Agent RL Environment
- 25 agents trained simultaneously
- DQN / Q-learning based architectures
- Sequential interaction recording
- Dynamic state transitions
- Configurable environment parameters

---

### 2. Rule Generation Module
The generative rule model dynamically modifies:
- Reward structure
- Cooperation incentives
- Mistake probability
- Reproduction rate
- Round number

This enables adaptive difficulty control and rule-space exploration.

---

### 3. Evaluation System
Tracked metrics include:
- Cooperation rate
- Gini coefficient
- Individual income distribution
- Skill vs difficulty analysis
- Rule difficulty embeddings (t-SNE)

All results are persisted and versioned.

---

### 4. Web Dashboard
- Visualizes experiment results
- Displays training curves
- Supports querying historical runs
- Designed for Azure deployment
- Separates presentation layer from ML logic

---

### 5. Cloud-Native Deployment
Designed for industrial-scale deployment on Azure:
- Azure Machine Learning (training jobs)
- Azure Data Lake Storage Gen2 (artifacts)
- Azure Container Apps (API service)
- Azure Static Web Apps (frontend)
- Azure Monitor + Log Analytics (observability)

---

## Running Locally

### 1. Install Dependencies
pip install -r requirements.txt

### 2. Train Agents
python ml/src/trust_evolution/Experiment.py

### 3. Launch Dashboard (Static Example)
Open in browser:
web/html/index.html

## Azure Deployment Overview
ML Training
Package training code as Azure ML Job
Attach GPU compute cluster (optional)
Store artifacts in ADLS Gen2
API Layer
Deploy as Azure Container App.
Example endpoints:

POST   /api/jobs
GET    /api/jobs/{id}
GET    /api/jobs/{id}/logs
GET    /api/jobs/{id}/metrics

## Web Frontend

Deploy homepage → Azure Static Web Apps

Deploy dashboard → Azure Static Web Apps


## Engineering Practices

Modular ML architecture
Config-driven experimentation
Clear separation of ML and Web layers
Cloud-native design
Scalable multi-agent simulation
Structured experiment outputs
Production-ready deployment mindset

## Why This Project Matters

This project demonstrates:
Reinforcement learning in multi-agent systems
Dynamic rule generation via generative models
Cloud deployment of ML workloads
Full-stack integration between ML and Web
Production-level experiment tracking & observability

It is not only a research environment —
it is designed as a deployable AI product.

## Author

Jiyao Pu
PhD in Computer Science (Machine Learning)
Durham University, UK

Focus areas:
Reinforcement Learning
Multi-Agent Systems
Generative AI
Rule Generation
Cloud-native ML Deployment