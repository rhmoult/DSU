python build_vector_index.py
uvicorn api:app --port 8000 &
sleep 100
garak -m rest -p rag_pii_probe -p latentinjection -p atkgen -p promptinject -p divergence -G ./garak_config.json --verbose
