.PHONY: up down build ingest logs db-shell clean

up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f

ingest:
	docker compose run --rm api python pipelines/ingest/download.py
	docker compose run --rm api python pipelines/ingest/load_raw.py

process:
	docker compose run --rm api python pipelines/ingest/process_interactions.py

embed:
	docker compose run --rm api python pipelines/embeddings/embed.py

verify:
	docker compose run --rm api python pipelines/quality/check_data.py

train:
	docker compose run --rm api python pipelines/models/train_cf.py
	docker compose run --rm api python pipelines/models/train_popular.py

features:
	docker compose run --rm api python pipelines/features/build_features.py

train-ranker:
	docker compose run --rm api python pipelines/models/train_ranker.py

evaluate:
	docker compose run --rm api python pipelines/evaluation/evaluate.py

score:
	docker compose run --rm api python pipelines/batch_score/daily_run.py

db-shell:
	docker compose exec db psql -U postgres -d recsys

clean:
	docker compose down -v
	rm -rf data/raw/* data/processed/* mlruns/*

