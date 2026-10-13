# Deep learning lab programs

Four short Jupyter notebooks written as college lab exercises: word embeddings, a dense classifier,
a CNN and an autoencoder, each in a single cell using Gensim or TensorFlow/Keras.

| Notebook | Program | Dataset | Result printed in the notebook |
|----------|---------|---------|--------------------------------|
| [`pgm_1.ipynb`](pgm_1.ipynb) | Word2Vec embeddings (Gensim): trains on a 3-sentence toy corpus, prints the 50-d vector for "learning" and its most similar words | inline toy corpus | similarity scores only (toy corpus, not meaningful) |
| [`pgm_2.ipynb`](pgm_2.ipynb) | Feed-forward neural network (Dense 16-12-8-3, softmax) for multi-class classification, 50 epochs | Iris (scikit-learn), 80/20 split | Test accuracy 100.00% on the 30-sample test split |
| [`pgm_3.ipynb`](pgm_3.ipynb) | CNN (2x Conv2D + MaxPooling, Dense 128, Dropout 0.5) for digit classification, 10 epochs | MNIST (Keras) | Test accuracy 99.26% |
| [`pgm_4.ipynb`](pgm_4.ipynb) | Single-layer autoencoder compressing 784-d images to 32-d, 5 epochs; plots originals vs reconstructions | MNIST (Keras) | Encoded shape (10000, 32) and a reconstruction plot |

Accuracies are from the saved outputs of a single run; they will vary slightly between runs.

## Earlier ML lab programs

[`ml_programs/`](ml_programs) holds four classical-ML lab notebooks that previously survived only as Jupyter checkpoint files:

| Notebook | Program | Dataset |
|----------|---------|---------|
| [`ml_pgm_1_find_s.ipynb`](ml_programs/ml_pgm_1_find_s.ipynb) | Find-S concept learning | inline "EnjoySport" table |
| [`ml_pgm_2_ripper_rules.ipynb`](ml_programs/ml_pgm_2_ripper_rules.ipynb) | Rule learning with RIPPER (`wittgenstein`) | Iris |
| [`ml_pgm_3_bagging_adaboost.ipynb`](ml_programs/ml_pgm_3_bagging_adaboost.ipynb) | Bagging vs AdaBoost over decision trees | Breast Cancer (scikit-learn) |
| [`ml_pgm_4_kmeans_iris.ipynb`](ml_programs/ml_pgm_4_kmeans_iris.ipynb) | K-Means clustering with standardized features | Iris |

Their dependencies (`pandas`, `scikit-learn`, `matplotlib`, `wittgenstein`) are included in `requirements.txt`.

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Iris ships with scikit-learn; MNIST is downloaded automatically by Keras on first run.
