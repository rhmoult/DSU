uvicorn proxy:app --port 8000 &
sleep 10
garak --model_type rest -G ./garak_config.json &
