.PHONY: install run clean

install:
	python -m pip install -e .

run:
	python -m fraud_guard

clean:
	rm -rf artifacts data/transactions.csv

