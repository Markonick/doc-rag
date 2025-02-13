.PHONY: qdrant streamlit

qdrant:
	docker run -p 6333:6333 -p 6334:6334 \
		-v $(PWD)/qdrant_storage:/qdrant/storage:z \
		qdrant/qdrant

ollama:
	ollama run llama3.2:1b

streamlit:
	poetry run streamlit run app.py
