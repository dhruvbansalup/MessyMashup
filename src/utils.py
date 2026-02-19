from .config import GENRES

def genre_to_idx(genre):
    return GENRES.index(genre)

def idx_to_genre(idx):
    return GENRES[idx]