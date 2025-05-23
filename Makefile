
setup: 
	sudo apt update
	sudo apt install uthash-dev
	sudo apt install libcjson-dev

download_gpt2:
	pip install -r requirements.txt
	python convert.py

compile:
	gcc map.c -lcjson -o map
	./map
	gcc run.c Tokenize.c timer.c -lcjson  -lm -o inference 

run:
	./inference "Hi" 10
