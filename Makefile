SHELL = /bin/bash

# Set variable
MESSAGE := "hello world"

# Use variable
greeting:
	@echo ${MESSAGE}



# Help
.PHONY: help
help:
	@echo "Commands: "
	@echo "venv : creates a virtual environment name venv."
	@echo "style : executes style formatting."
	@echo "clean : cleans all unnecessary files."

# Styling
.PHONY: style
style:
	black .
	flake8
	python3 -m isort .

# Clean
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".trash" | xargs rm -rf
	rm -f .coverage

# Test
.PHONY: test
test:
	pytest -m "not training"
	cd tests && great_expectations checkpoint run projects
	cd tests && great_expectations checkpoint run tags
	cd tests && great_expectations checkpoint run labeled_projects


# DVC
.PHONY: DVC
dvc :
	dvc add data/labeled_projects.csv
	dvc add data/projects.csv
	dvc add data/tags.csv
	dvc push

# Environment
venv :
	python3 -m venv venv
	source venv/bin/activate && \
	python3 -m pip install pip setuptools wheel && \
	python3 -m pip install -e .
	pre-commit install && \
    pre-commit autoupdate
