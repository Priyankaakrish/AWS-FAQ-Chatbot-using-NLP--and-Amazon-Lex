# AWS FAQ Chatbot — common workflows
DATA    ?= data/bitext.csv
BACKEND ?= lsa
ART     ?= artifacts
REPORTS ?= reports

.PHONY: help install sample train evaluate analytics lex test all demo clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:   ## Install Python dependencies
	pip install -r requirements.txt

sample:    ## Generate synthetic Bitext-shaped data (no Kaggle download needed)
	python scripts/make_sample_data.py data/sample_bitext.csv

train:     ## Fit intent classifier + FAQ index
	python -m src.train --data $(DATA) --backend $(BACKEND) --out $(ART)

evaluate:  ## Full metrics suite into reports/
	python -m src.evaluate --artifacts $(ART) --out $(REPORTS) --ood data/ood_questions.txt

analytics: ## Chatbot analytics report from local predictions
	python scripts/analytics.py --source $(REPORTS)/predictions.csv --out $(REPORTS)/analytics.json

lex:       ## Generate the Lex V2 bot definition
	python lex/build_lex_bot.py --data $(DATA) --out lex/bot_definition.json

test:      ## Run the test suite
	pytest -q tests/

demo:      ## Interactive REPL
	python -m src.cli --artifacts $(ART)

all: train evaluate analytics lex  ## train -> evaluate -> analytics -> lex

# End-to-end run against the synthetic sample, for a clean checkout
smoke: sample
	$(MAKE) all DATA=data/sample_bitext.csv
	$(MAKE) test

clean:
	rm -rf $(ART)/*.joblib $(ART)/*.csv $(REPORTS)/* layer/ *.zip
	find . -name __pycache__ -type d -exec rm -rf {} +
