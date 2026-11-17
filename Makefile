PY ?= python

.PHONY: all data analysis report test clean

all: analysis report

data:            ## download and cache prices (needs internet)
	$(PY) scripts/01_download_data.py

analysis:        ## run every analysis step on the cached data
	$(PY) scripts/run_all.py

report:          ## compile the LaTeX report
	cd report && pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex

test:            ## unit tests (synthetic data, no internet)
	$(PY) -m pytest -q

clean:
	rm -rf data/processed results/figures/* results/tables/* results/numbers/* results/graphs/* report/generated/*
