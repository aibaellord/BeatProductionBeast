install:
	pip install -r requirements.txt
	pip install -r dev-requirements.txt || true

lint:
	flake8 src/

format:
	black src/
	isort src/

test:
	pytest --cov=src

dev:
	uvicorn src.api:app --reload

docker-build:
	docker build -t beatproductionbeast .

docker-run:
	docker run -p 8000:8000 beatproductionbeast
