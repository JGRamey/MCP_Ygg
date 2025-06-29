# MCP Server Makefile

.PHONY: install dev test lint format clean docker k8s

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

dev:
	pip install -r requirements.txt
	pre-commit install

test:
	python scripts/run_tests.py

lint:
	python tests/lint/lint_project.py

lint-fix:
	python tests/lint/lint_project.py --fix

lint-parallel:
	python tests/lint/lint_project.py --parallel

lint-sequential:
	python tests/lint/lint_project.py --sequential

lint-report:
	python tests/lint/lint_project.py --output lint_report.txt

lint-individual:
	flake8 agents/ api/ dashboard/ scripts/ tests/
	black --check agents/ api/ dashboard/ scripts/ tests/
	isort --check-only agents/ api/ dashboard/ scripts/ tests/
	mypy agents/ api/ dashboard/ scripts/
	pylint agents/ api/ dashboard/ scripts/
	bandit -r agents/ api/ dashboard/ scripts/
	ruff check agents/ api/ dashboard/ scripts/ tests/

setup-dev:
	pip install -r tests/lint/requirements-dev.txt
	pre-commit install
	python -m spacy download en_core_web_sm

setup-lint:
	python tests/lint/setup_linting.py

format:
	black agents/ api/ tests/
	isort agents/ api/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/

docker:
	docker-compose up -d

docker-build:
	docker build -t mcp-server:latest .

k8s-deploy:
	kubectl apply -f k8s/

k8s-delete:
	kubectl delete -f k8s/

init:
	python scripts/initialize_system.py

run-api:
	python -m api.main

run-dashboard:
	streamlit run dashboard/app.py

help:
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  dev          - Set up development environment"
	@echo "  test         - Run all tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  clean        - Clean temporary files"
	@echo "  docker       - Start Docker services"
	@echo "  k8s-deploy   - Deploy to Kubernetes"
	@echo "  init         - Initialize system"
	@echo "  run-api      - Start API server"
	@echo "  run-dashboard - Start dashboard"
