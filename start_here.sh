python build_vector_index.py
uvicorn api:app --port 8000 &
sleep 100
garak --model_type rest -G ./garak_config.json &
