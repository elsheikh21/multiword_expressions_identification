import multiprocessing
from gensim.models import Word2Vec


def train_pos2vec(training_data, w, embed_size, lr, epochs):
    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=5, window=w,
                         size=embed_size, sample=1e-3,
                         alpha=lr, min_alpha=0.0007,
                         negative=20, workers=cores-1)
    w2v_model.build_vocab(training_data, progress_per=10000)
    print(w2v_model)
    w2v_model.train(training_data, total_examples=w2v_model.corpus_count,
                    epochs=epochs, report_delay=1)
    return w2v_model


def save_pos_embeddings(w2v_model, path_to_save):
    pos_embeds = [token for token in w2v_model.wv.vocab]

    with open(path_to_save, "w+") as f:
        for key in pos_embeds:
            f.write(f"{key} ")
            for el in w2v_model[key]:
                f.write(f"{el} ")
            f.write("\n")
