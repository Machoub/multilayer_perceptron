PY ?= python3

.PHONY: all install split train predict test clean

all: split train predict

install:
	$(PY) -m pip install -r requirements.txt

split:
	$(PY) split.py --data data.csv --val-size 0.2 --seed 42

train:
	$(PY) train.py --layer 24 24 --epochs 84 --batch_size 8 --learning_rate 0.0314

predict:
	$(PY) predict.py --model saved_model.npy --data data_valid.csv

test:
	$(PY) tests/gradient_check.py

clean:
	rm -f saved_model.npy learning_curves.png data_features.png \
	      data_train.csv data_valid.csv
