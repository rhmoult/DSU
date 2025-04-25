python build_vector_index.py
uvicorn api:app --port 8000 &
sleep 100
garak -m rest -p pii -G ./garak_config.json --verbose
for file in $(ls -ltr ~/.local/share/garak/garak_runs/ | tail -2 | awk -F ' ' '{print }'); do cp $(find ~ -type f -name $file) logs/pii_$file 2>/dev/null; done
garak -m rest -p latentinjection -G ./garak_config.json --verbose
for file in $(ls -ltr ~/.local/share/garak/garak_runs/ | tail -2 | awk -F ' ' '{print }'); do cp $(find ~ -type f -name $file) logs/latentinjection_$file 2>/dev/null; done
garak -m rest -p atkgen -G ./garak_config.json --verbose
for file in $(ls -ltr ~/.local/share/garak/garak_runs/ | tail -2 | awk -F ' ' '{print }'); do cp $(find ~ -type f -name $file) logs/atkgen_$file 2>/dev/null; done
garak -m rest -p promptinject -G ./garak_config.json --verbose
for file in $(ls -ltr ~/.local/share/garak/garak_runs/ | tail -2 | awk -F ' ' '{print }'); do cp $(find ~ -type f -name $file) logs/promptinject_$file 2>/dev/null; done
garak -m rest -p divergence -G ./garak_config.json --verbose
for file in $(ls -ltr ~/.local/share/garak/garak_runs/ | tail -2 | awk -F ' ' '{print }'); do cp $(find ~ -type f -name $file) logs/divergence_$file 2>/dev/null; done
