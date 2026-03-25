# Flavinator - AI Food Guessing Game

Think of any dish from anywhere in the world. Flavinator will guess it.

## Live Demo
https://mathurharshal2006-flavinator.hf.space

## How it works
- Naive Bayes classifier built with PyTorch tracks dish probabilities
- Decision Tree selects the most informative question using Information Gain
- Average 3-4 questions needed to identify 1 dish from 40

## Tech Stack
- PyTorch, Pandas, Scikit-learn
- FastAPI, Streamlit
- Docker, GitHub Actions CI/CD
- MLflow experiment tracking

## Run locally
git clone https://github.com/mathurharshal2006/flavinator.git
cd flavinator
docker compose up --build

## Project Structure
src/data/dishes.py          - 40 dish knowledge base
src/models/naive_bayes.py   - Naive Bayes with PyTorch tensors
src/models/decision_tree.py - Information Gain question selector
src/game/engine.py          - combines both algorithms
src/api/main.py             - FastAPI REST API
src/app.py                  - Streamlit game UI
tests/test_engine.py        - 20 automated tests
Dockerfile                  - containerization
docker-compose.yml          - multi service deployment
.github/workflows/ci.yml    - CI/CD pipeline
monitoring/dashboard.py     - MLflow monitoring dashboard