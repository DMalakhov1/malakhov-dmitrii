"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import lightgbm as lgb
import re
from collections import Counter, defaultdict
import warnings
import gc
from tqdm import tqdm
from joblib import Parallel, delayed
import time
import os
def create_submission(predictions, test_ids):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """
    # Создать пандас таблицу submission
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    submission = pd.DataFrame({'id': test_ids, 'prediction': predictions})
    submission = submission.sort_values('id').reset_index(drop=True)
    submission.to_csv(submission_path, index=False)
    print(f"Submission файл сохранен: {submission_path}")
    return submission_path
def main():
    """
    Главная функция программы
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    # Дополнительные библиотеки
    try:
        from rank_bm25 import BM25Okapi, BM25Plus, BM25L
        BM25_AVAILABLE = True
    except ImportError:
        print("pip install rank-bm25, котик внимательней")
        BM25_AVAILABLE = False
    try:
        import Levenshtein
        LEVENSHTEIN_AVAILABLE = True
    except ImportError:
        print("pip install python-Levenshtein, котик внимательней")
        LEVENSHTEIN_AVAILABLE = False
    try:
        from sentence_transformers import SentenceTransformer, CrossEncoder
        SENTENCE_TRANSFORMERS_AVAILABLE = True
    except ImportError:
        print("pip install sentence-transformers, котик внимательней")
        SENTENCE_TRANSFORMERS_AVAILABLE = False
    try:
        from catboost import CatBoostRanker, Pool
        CATBOOST_AVAILABLE = True
    except ImportError:
        print("pip install catboost, котик внимательней")
        CATBOOST_AVAILABLE = False
    try:
        import xgboost as xgb
        XGBOOST_AVAILABLE = True
    except ImportError:
        print("pip install xgboost, котик внимательней")
        XGBOOST_AVAILABLE = False
    try:
        from gensim.models import Word2Vec
        GENSIM_AVAILABLE = True
    except ImportError:
        print("pip install gensim, котик внимательней")
        GENSIM_AVAILABLE = False
    try:
        from scipy.spatial.distance import cosine as cosine_dist
        from scipy.stats import rankdata
        SCIPY_AVAILABLE = True
    except ImportError:
        print("pip install scipy, котик внимательней")
        SCIPY_AVAILABLE = False
    warnings.filterwarnings('ignore')
    # КОНФИГУРАЦИЯ
    CONFIG = {
        'SEED': 993,
        'STOP_WORDS': set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                           'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be',
                           'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                           'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those']),
        'SVD_COMPONENTS': 250,
        'TFIDF_MAX_FEATURES': 50000,
        'BATCH_SIZE': 64,
        'N_JOBS': -1,
        'BI_ENCODER_MODEL': 'BAAI/bge-large-en-v1.5',
        'CROSS_ENCODER_MODEL': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
        'USE_CROSS_ENCODER': True,
        'CROSS_ENCODER_TOP_K': 50,
        'USE_CATBOOST': True,
        'USE_XGBOOST': True,
        'USE_WORD2VEC': True,
        'USE_VALIDATION': True,
        'VALIDATION_SIZE': 0.15,
        'FEATURE_SELECTION_K': 400,
        'USE_NOISE_CLEANING': True,
        'USE_ENTITY_FEATURES': True,
        'USE_INTERACTION_FEATURES': True,
    }
    np.random.seed(CONFIG['SEED'])
    # УТИЛИТЫ
    def optimize_dtypes(df):
        """Оптимизация памяти"""
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        for col in df.select_dtypes(include=['int64']).columns:
            if df[col].min() >= 0:
                if df[col].max() < 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() < 65535:
                    df[col] = df[col].astype('uint16')
        return df
    def get_column_name(df, possible_names):
        """Найти колонку по возможным именам"""
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    # УЛУЧШЕННАЯ ОЧИСТКА
    def fix_repeating_numbers(text):
        """Исправляет 613613 -> 60, 527527 -> 52"""
        if not text or pd.isna(text):
            return text
        text = str(text)
        pattern = r'(\d{2,})(\1)'
        def replacer(match):
            full = match.group(0)
            first_part = match.group(1)
            if len(full) % 2 == 0 and full[:len(full)//2] == full[len(full)//2:]:
                return first_part
            return full
        return re.sub(pattern, replacer, text)
    def correct_common_typos(text):
        """Исправляет частые опечатки"""
        if not text:
            return ""
        typo_map = {
            'fuplicate': 'duplicate', 'ps104': 'ps4', 'ps105': 'ps5',
            'iph0ne': 'iphone', 'samung': 'samsung', 'lapt0p': 'laptop',
            'w0men': 'women', 'men5': 'mens', 'wom3n': 'women'
        }
        words = text.split()
        return ' '.join([typo_map.get(w, w) for w in words])
    def clean_text(text, remove_stopwords=True, fix_numbers=True):
        """Прокачанная очистка текста"""
        if pd.isna(text) or text is None:
            return ""
        text = str(text).lower()
        if fix_numbers and CONFIG['USE_NOISE_CLEANING']:
            text = fix_repeating_numbers(text)
        text = re.sub(r'#\S+', '', text)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = correct_common_typos(text)
        if remove_stopwords:
            tokens = [w for w in text.split() if w not in CONFIG['STOP_WORDS'] and len(w) > 2]
        else:
            tokens = [w for w in text.split() if len(w) > 2]
        return ' '.join(tokens)
    # НОВЫЕ ENTITY FEATURES
    def compute_entity_features(df):
        """Brand и Color matching"""
        if not CONFIG['USE_ENTITY_FEATURES']:
            return pd.DataFrame({'brand_match': [0]*len(df), 'color_match': [0]*len(df)}, index=df.index)
        features = pd.DataFrame(index=df.index)
        COLORS = {'red', 'blue', 'green', 'yellow', 'black', 'white', 'pink', 'purple',
                  'orange', 'brown', 'grey', 'gray', 'silver', 'gold'}
        def brand_score(row):
            if pd.isna(row['brand']) or 'unknown' in str(row['brand']).lower():
                return 0.0
            brand = str(row['brand']).lower()
            query = str(row['query_original']).lower()
            if brand in query:
                return 1.0
            if LEVENSHTEIN_AVAILABLE and brand:
                return Levenshtein.jaro_winkler(query, brand)
            return 0.0
        features['brand_match'] = df.apply(brand_score, axis=1)
        def color_score(row):
            query = str(row['query_original']).lower()
            q_colors = [c for c in COLORS if c in query]
            if not q_colors or pd.isna(row.get('color')):
                return 0.0
            color = str(row['color']).lower()
            return 1.0 if any(c in color for c in q_colors) else 0.0
        features['color_match'] = df.apply(color_score, axis=1)
        return features
    # НОВЫЕ NOISE FEATURES
    def compute_noise_features(df):
        """Признаки для работы с шумом"""
        features = pd.DataFrame(index=df.index)
        features['query_num_count'] = df['query_original'].apply(lambda x: len(re.findall(r'\d+', str(x))))
        features['title_num_count'] = df['title_original'].apply(lambda x: len(re.findall(r'\d+', str(x))))
        def num_overlap(row):
            q_nums = set(re.findall(r'\d+', str(row['query_original'])))
            t_nums = set(re.findall(r'\d+', str(row['title_original'])))
            return len(q_nums & t_nums) / len(q_nums) if q_nums else 0.0
        features['number_overlap'] = df.apply(num_overlap, axis=1)
        features['query_has_hashtag'] = df['query'].apply(lambda x: 1 if '#' in str(x) else 0)
        return features
    # ФУНКЦИИ ПРИЗНАКОВ
    def compute_idf(documents):
        """Вычисление IDF"""
        N = len(documents)
        doc_freq = defaultdict(int)
        for doc in documents:
            for word in set(doc.split()):
                doc_freq[word] += 1
        return {word: np.log((N - freq + 0.5) / (freq + 0.5) + 1.0) for word, freq in doc_freq.items()}
    def compute_bm25_features_advanced(df):
        """BM25 (Okapi, Plus, L)"""
        if not BM25_AVAILABLE:
            return {f'bm25_{v}': [0.0]*len(df) for v in ['okapi_title', 'plus_title', 'l_title', 'okapi_product']}
        results = {'bm25_okapi_title': [], 'bm25_plus_title': [], 'bm25_l_title': [], 'bm25_okapi_product': []}
        for qid in tqdm(df['query_id'].unique(), desc="BM25"):
            mask = df['query_id'] == qid
            query_tokens = df.loc[mask, 'query_clean'].iloc[0].split()
            title_docs = [doc.split() for doc in df.loc[mask, 'title_clean'].fillna('')]
            product_docs = [doc.split() for doc in df.loc[mask, 'product_text'].fillna('')]
            results['bm25_okapi_title'].extend(BM25Okapi(title_docs).get_scores(query_tokens))
            results['bm25_plus_title'].extend(BM25Plus(title_docs).get_scores(query_tokens))
            results['bm25_l_title'].extend(BM25L(title_docs).get_scores(query_tokens))
            results['bm25_okapi_product'].extend(BM25Okapi(product_docs).get_scores(query_tokens))
        return results
    def compute_tfidf_cosine_similarity(train_df, test_df):
        """TF-IDF Cosine (Оптимизировано)"""
        tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), min_df=2)
        all_texts = pd.concat([train_df['query_clean'], train_df['product_text'],
                               test_df['query_clean'], test_df['product_text']]).fillna('')
        tfidf_vectorizer.fit(all_texts)
        def batch_cosine(query_vec, product_vec, batch_size=1000):
            results = []
            for i in tqdm(range(0, query_vec.shape[0], batch_size), desc="TF-IDF cosine"):
                batch_q = query_vec[i:i+batch_size]
                batch_p = product_vec[i:i+batch_size]
                sims = np.array([cosine_similarity(batch_q[j], batch_p[j])[0, 0] for j in range(batch_q.shape[0])])
                results.extend(sims)
            return np.array(results)
        train_query_vec = tfidf_vectorizer.transform(train_df['query_clean'].fillna(''))
        train_product_vec = tfidf_vectorizer.transform(train_df['product_text'].fillna(''))
        test_query_vec = tfidf_vectorizer.transform(test_df['query_clean'].fillna(''))
        test_product_vec = tfidf_vectorizer.transform(test_df['product_text'].fillna(''))
        train_cosine = batch_cosine(train_query_vec, train_product_vec)
        test_cosine = batch_cosine(test_query_vec, test_product_vec)
        return train_cosine, test_cosine
    def train_word2vec_model(train_df, test_df, vector_size=100, window=5):
        """Обучение Word2Vec"""
        if not GENSIM_AVAILABLE:
            return None
        all_texts = pd.concat([train_df['query_clean'], train_df['product_text'],
                               test_df['query_clean'], test_df['product_text']]).fillna('')
        sentences = [text.split() for text in all_texts if text]
        print(f"Training Word2Vec on {len(sentences)} sentences...")
        model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window,
                         min_count=2, workers=4, sg=1, epochs=10, seed=CONFIG['SEED'])
        return model
    def compute_word2vec_features(df, w2v_model):
        """Word2Vec признаки"""
        if w2v_model is None:
            return pd.DataFrame({'w2v_cosine': [0.0]*len(df), 'w2v_query_coverage': [0.0]*len(df),
                                 'w2v_title_coverage': [0.0]*len(df)})
        features = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Word2Vec"):
            q_words = [w for w in row['query_clean'].split() if w in w2v_model.wv]
            t_words = [w for w in row['title_clean'].split() if w in w2v_model.wv]
            q_vec = np.mean([w2v_model.wv[w] for w in q_words], axis=0) if q_words else np.zeros(w2v_model.vector_size)
            t_vec = np.mean([w2v_model.wv[w] for w in t_words], axis=0) if t_words else np.zeros(w2v_model.vector_size)
            cosine_sim = 1 - cosine_dist(q_vec, t_vec) if SCIPY_AVAILABLE and len(q_words) > 0 and len(t_words) > 0 else 0.0
            features.append({'w2v_cosine': cosine_sim, 'w2v_query_coverage': len(q_words) / max(len(row['query_clean'].split()), 1),
                             'w2v_title_coverage': len(t_words) / max(len(row['title_clean'].split()), 1)})
        return pd.DataFrame(features)
    def compute_advanced_features(df, idf_dict):
        """Позиционные + IDF признаки"""
        def extract_features(row):
            q_words = row['query_clean'].split()
            t_words = row['title_clean'].split()
            if not q_words or not t_words:
                return {k: 0.0 for k in ['avg_position', 'first_position', 'position_variance', 'idf_weighted_overlap', 'avg_query_idf']}
            positions = [i for i, word in enumerate(t_words) if word in q_words]
            if positions:
                avg_pos = np.mean(positions) / (len(t_words) + 1)
                first_pos = positions[0] / (len(t_words) + 1)
                pos_var = np.var(positions) / (len(t_words) + 1) if len(positions) > 1 else 0.0
            else:
                avg_pos = first_pos = 1.0
                pos_var = 0.0
            overlap_words = set(q_words) & set(t_words)
            idf_weighted = sum(idf_dict.get(w, 0) for w in overlap_words)
            total_q_idf = sum(idf_dict.get(w, 0) for w in q_words)
            idf_overlap_norm = idf_weighted / total_q_idf if total_q_idf > 0 else 0.0
            avg_q_idf = np.mean([idf_dict.get(w, 0) for w in q_words])
            return {'avg_position': avg_pos, 'first_position': first_pos, 'position_variance': pos_var,
                    'idf_weighted_overlap': idf_overlap_norm, 'avg_query_idf': avg_q_idf}
        return df.apply(extract_features, axis=1, result_type='expand')
    def compute_ngram_features(df):
        """N-gram признаки"""
        def ngram_match(row):
            q_words = row['query_clean'].split()
            t_words = row['title_clean'].split()
            if len(q_words) < 2:
                return [0, 0, 0, 0]
            q_bigrams = set(tuple(q_words[i:i+2]) for i in range(len(q_words)-1))
            t_bigrams = set(tuple(t_words[i:i+2]) for i in range(len(t_words)-1)) if len(t_words) > 1 else set()
            bigram_match = len(q_bigrams & t_bigrams) / len(q_bigrams) if q_bigrams else 0
            trigram_match = 0
            if len(q_words) >= 3:
                q_trigrams = set(tuple(q_words[i:i+3]) for i in range(len(q_words)-2))
                t_trigrams = set(tuple(t_words[i:i+3]) for i in range(len(t_words)-2)) if len(t_words) >= 3 else set()
                trigram_match = len(q_trigrams & t_trigrams) / len(q_trigrams) if q_trigrams else 0
            q_clean = row['query_clean']
            t_clean = row['title_clean']
            q_char_bi = set(q_clean[i:i+2] for i in range(len(q_clean)-1)) if len(q_clean) >= 2 else set()
            t_char_bi = set(t_clean[i:i+2] for i in range(len(t_clean)-1)) if len(t_clean) >= 2 else set()
            char_bi_match = len(q_char_bi & t_char_bi) / len(q_char_bi) if q_char_bi else 0
            q_char_tri = set(q_clean[i:i+3] for i in range(len(q_clean)-2)) if len(q_clean) >= 3 else set()
            t_char_tri = set(t_clean[i:i+3] for i in range(len(t_clean)-2)) if len(t_clean) >= 3 else set()
            char_tri_match = len(q_char_tri & t_char_tri) / len(q_char_tri) if q_char_tri else 0
            return [bigram_match, trigram_match, char_bi_match, char_tri_match]
        ngram_results = df.apply(ngram_match, axis=1, result_type='expand')
        ngram_results.columns = ['bigram_match', 'trigram_match', 'char_bigram_match', 'char_trigram_match']
        return ngram_results
    def compute_text_features_vectorized(df):
        """Базовые текстовые признаки"""
        features = pd.DataFrame(index=df.index)
        features['query_len'] = df['query_clean'].fillna('').str.split().str.len()
        features['title_len'] = df['title_clean'].fillna('').str.split().str.len()
        features['product_len'] = df['product_text'].fillna('').str.split().str.len()
        q_words = df['query_clean'].fillna('').str.split().apply(set)
        t_words = df['title_clean'].fillna('').str.split().apply(set)
        q_t_intersection = q_words.combine(t_words, lambda a, b: len(a & b))
        q_t_union = q_words.combine(t_words, lambda a, b: len(a | b))
        features['jaccard_title'] = q_t_intersection / q_t_union.replace(0, 1)
        features['title_match_ratio'] = q_t_intersection / q_words.str.len().replace(0, 1)
        features['all_words_in_title'] = (q_t_intersection == q_words.str.len()).astype(int)
        features['dice_coef'] = (2 * q_t_intersection) / (q_words.str.len() + t_words.str.len()).replace(0, 1)
        features['overlap_coef'] = q_t_intersection / q_words.combine(t_words, lambda a, b: min(len(a), len(b)) if a and b else 1)
        features['len_ratio'] = features['query_len'] / (features['title_len'] + 1)
        features['has_brand'] = (~df['brand'].isna()).astype(int)
        features['exact_match'] = (df['query_original'].fillna('') == df['title_original'].fillna('')).astype(int)
        features['query_specificity'] = df['query_clean'].str.split().apply(lambda x: len(set(x)) / max(len(x), 1))
        features['query_avg_word_len'] = df['query_clean'].apply(lambda x: np.mean([len(w) for w in x.split()]) if x else 0)
        return features
    def compute_similarity_parallel(df, n_jobs=-1):
        """Levenshtein similarity"""
        if not LEVENSHTEIN_AVAILABLE:
            return pd.DataFrame({'jaro_winkler': [0.0]*len(df), 'edit_distance_norm': [0.0]*len(df)}, index=df.index)
        def process_batch(batch_df):
            results = []
            for _, row in batch_df.iterrows():
                q = str(row['query_clean'])[:100]
                t = str(row['title_clean'])[:100]
                jaro = Levenshtein.jaro_winkler(q, t) if q and t else 0.0
                edit_dist = Levenshtein.distance(q, t) if q and t else 0
                norm_edit = edit_dist / max(len(q), len(t), 1)
                results.append({'jaro_winkler': jaro, 'edit_distance_norm': norm_edit})
            return pd.DataFrame(results)
        batch_size = max(len(df) // 10, 1000)
        batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]
        results = Parallel(n_jobs=n_jobs, backend='loky')(delayed(process_batch)(batch) for batch in tqdm(batches, desc="Similarity"))
        return pd.concat(results, ignore_index=True)
    def compute_pairwise_features(df, feature_cols):
        """Pairwise сравнения"""
        pairwise_features = pd.DataFrame(index=df.index)
        for col in feature_cols:
            if col not in df.columns:
                continue
            df[f'{col}_rank_norm'] = df.groupby('query_id')[col].rank(ascending=False, method='dense')
            df[f'{col}_rank_norm'] = df.groupby('query_id')[f'{col}_rank_norm'].transform(lambda x: x / max(x.max(), 1))
            df[f'{col}_diff_from_max'] = df.groupby('query_id')[col].transform(lambda x: x.max() - x)
            pairwise_features[f'{col}_rank_norm'] = df[f'{col}_rank_norm']
            pairwise_features[f'{col}_diff_from_max'] = df[f'{col}_diff_from_max']
        return pairwise_features
    def reciprocal_rank_fusion(df, score_columns, k=60):
        """Reciprocal Rank Fusion"""
        rrf_scores = np.zeros(len(df))
        for col in score_columns:
            if col not in df.columns:
                continue
            df[f'{col}_rrf_rank'] = df.groupby('query_id')[col].rank(ascending=False, method='first')
            rrf_scores += 1.0 / (k + df[f'{col}_rrf_rank'].values)
        return rrf_scores
    def calibrate_predictions(predictions, query_ids, method='minmax'):
        """Калибровка предсказаний"""
        if not SCIPY_AVAILABLE:
            return predictions
        calibrated = np.zeros_like(predictions)
        unique_qids = np.unique(query_ids)
        for qid in unique_qids:
            mask = query_ids == qid
            qid_preds = predictions[mask]
            if method == 'minmax':
                qid_min, qid_max = qid_preds.min(), qid_preds.max()
                calibrated[mask] = (qid_preds - qid_min) / (qid_max - qid_min) if qid_max > qid_min else 0.5
            elif method == 'rank':
                calibrated[mask] = rankdata(-qid_preds, method='ordinal') / len(qid_preds)
        return calibrated
    # ЗАГРУЗКА ДАННЫХ
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    test_ids_original = test_df['id'].copy()
    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")
    # ПРЕПРОЦЕССИНГ
    for df in [train_df, test_df]:
        title_col = get_column_name(df, ['producttitle', 'product_title'])
        desc_col = get_column_name(df, ['productdescription', 'product_description'])
        bullet_col = get_column_name(df, ['productbulletpoint', 'product_bulletpoint'])
        brand_col = get_column_name(df, ['productbrand', 'product_brand'])
        color_col = get_column_name(df, ['productcolor', 'product_color'])
        qid_col = get_column_name(df, ['queryid', 'query_id'])
        df['query_clean'] = df['query'].apply(lambda x: clean_text(x, remove_stopwords=True, fix_numbers=True))
        df['query_original'] = df['query'].apply(lambda x: clean_text(x, remove_stopwords=False, fix_numbers=True))
        df['title_clean'] = df[title_col].apply(lambda x: clean_text(x, remove_stopwords=True, fix_numbers=True)) if title_col else ""
        df['title_original'] = df[title_col].apply(lambda x: clean_text(x, remove_stopwords=False, fix_numbers=True)) if title_col else ""
        df['desc_clean'] = df[desc_col].apply(lambda x: clean_text(x, remove_stopwords=True, fix_numbers=True)) if desc_col else ""
        df['bullet_clean'] = df[bullet_col].apply(lambda x: clean_text(x, remove_stopwords=True, fix_numbers=True)) if bullet_col else ""
        df['brand'] = df[brand_col] if brand_col else None
        df['color'] = df[color_col] if color_col else None
        df['query_id'] = df[qid_col]
        df['product_text'] = (df['title_clean'].fillna('') + ' ' + df['desc_clean'].fillna('') + ' ' + df['bullet_clean'].fillna('')).str.strip().fillna("")
    print(" Препроцессинг готов")
    # BM25
    train_bm25 = compute_bm25_features_advanced(train_df)
    test_bm25 = compute_bm25_features_advanced(test_df)
    for key in train_bm25.keys():
        train_df[key] = train_bm25[key]
        test_df[key] = test_bm25[key]
    # IDF
    idf_dict = compute_idf(pd.concat([train_df['product_text'], test_df['product_text']]).tolist())
    train_advanced = compute_advanced_features(train_df, idf_dict)
    test_advanced = compute_advanced_features(test_df, idf_dict)
    # TF-IDF COSINE
    train_tfidf_cosine, test_tfidf_cosine = compute_tfidf_cosine_similarity(train_df, test_df)
    train_df['tfidf_cosine'] = train_tfidf_cosine
    test_df['tfidf_cosine'] = test_tfidf_cosine
    # WORD2VEC
    if CONFIG['USE_WORD2VEC']:
        w2v_model = train_word2vec_model(train_df, test_df, vector_size=100, window=5)
        train_w2v_features = compute_word2vec_features(train_df, w2v_model)
        test_w2v_features = compute_word2vec_features(test_df, w2v_model)
    else:
        train_w2v_features = pd.DataFrame({'w2v_cosine': [0.0]*len(train_df), 'w2v_query_coverage': [0.0]*len(train_df), 'w2v_title_coverage': [0.0]*len(train_df)})
        test_w2v_features = pd.DataFrame({'w2v_cosine': [0.0]*len(test_df), 'w2v_query_coverage': [0.0]*len(test_df), 'w2v_title_coverage': [0.0]*len(test_df)})
    # N-GRAM
    train_ngram = compute_ngram_features(train_df)
    test_ngram = compute_ngram_features(test_df)
    # БАЗОВЫЕ ПРИЗНАКИ
    train_text_features = compute_text_features_vectorized(train_df)
    test_text_features = compute_text_features_vectorized(test_df)
    # SIMILARITY
    train_similarity = compute_similarity_parallel(train_df, n_jobs=CONFIG['N_JOBS'])
    test_similarity = compute_similarity_parallel(test_df, n_jobs=CONFIG['N_JOBS'])
    # ENTITY FEATURES
    train_entity = compute_entity_features(train_df)
    test_entity = compute_entity_features(test_df)
    # NOISE FEATURES
    train_noise = compute_noise_features(train_df)
    test_noise = compute_noise_features(test_df)
    # TF-IDF + SVD
    all_texts = pd.concat([(train_df['query_clean'].fillna('') + ' ' + train_df['product_text'].fillna('')),
                           (test_df['query_clean'].fillna('') + ' ' + test_df['product_text'].fillna(''))])
    tfidf = TfidfVectorizer(max_features=CONFIG['TFIDF_MAX_FEATURES'], ngram_range=(1,3), min_df=2, dtype=np.float32)
    tfidf_matrix = tfidf.fit_transform(all_texts)
    svd = TruncatedSVD(n_components=CONFIG['SVD_COMPONENTS'], algorithm='randomized', random_state=CONFIG['SEED'])
    svd_features = svd.fit_transform(tfidf_matrix).astype(np.float32)
    train_svd_df = pd.DataFrame(svd_features[:len(train_df)], columns=[f'svd_{i}' for i in range(CONFIG['SVD_COMPONENTS'])])
    test_svd_df = pd.DataFrame(svd_features[len(train_df):], columns=[f'svd_{i}' for i in range(CONFIG['SVD_COMPONENTS'])])
    print(f" SVD variance: {svd.explained_variance_ratio_.sum():.3f}")
    # QUERY STATS
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    query_stats = all_df.groupby('query_id').agg(
        query_avg_title_len=('title_clean', lambda x: x.str.split().apply(len).mean()),
        query_std_title_len=('title_clean', lambda x: x.str.split().apply(len).std()),
        query_item_count=('query_id', 'size')
    ).reset_index()
    query_stats['query_std_title_len'] = query_stats['query_std_title_len'].fillna(0)
    train_df = train_df.merge(query_stats, on='query_id', how='left')
    test_df = test_df.merge(query_stats, on='query_id', how='left')
    # TARGET ENCODING
    if 'relevance' in train_df.columns:
        train_df['query_id_target_enc'] = 0
        gkf = GroupKFold(n_splits=5)
        for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, train_df['relevance'], train_df['query_id'])):
            X_train_fold = train_df.iloc[train_idx]
            X_val_fold = train_df.iloc[val_idx]
            agg = X_train_fold.groupby('query_id')['relevance'].agg(['mean', 'count'])
            global_mean = X_train_fold['relevance'].mean()
            smooth_target = (agg['mean'] * agg['count'] + global_mean * 10) / (agg['count'] + 10)
            train_df.loc[val_idx, 'query_id_target_enc'] = X_val_fold['query_id'].map(smooth_target).fillna(global_mean)
        agg = train_df.groupby('query_id')['relevance'].agg(['mean', 'count'])
        global_mean = train_df['relevance'].mean()
        smooth_target = (agg['mean'] * agg['count'] + global_mean * 10) / (agg['count'] + 10)
        test_df['query_id_target_enc'] = test_df['query_id'].map(smooth_target).fillna(global_mean)
    # BGE EMBEDDINGS
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            embedder = SentenceTransformer(CONFIG['BI_ENCODER_MODEL'], device='cpu')
            def add_bge_instruction(text, is_query=True):
                if 'bge' in CONFIG['BI_ENCODER_MODEL'].lower() and is_query:
                    return f"Represent this sentence for searching relevant passages: {text}"
                return text
            unique_queries = np.unique(np.concatenate([train_df['query_clean'].unique(), test_df['query_clean'].unique()]))
            query_embeddings_dict = {}
            for i in tqdm(range(0, len(unique_queries), CONFIG['BATCH_SIZE']), desc="Queries"):
                batch = unique_queries[i:i+CONFIG['BATCH_SIZE']]
                batch_with_inst = [add_bge_instruction(q, True) for q in batch]
                embeddings = embedder.encode(batch_with_inst, batch_size=CONFIG['BATCH_SIZE'], show_progress_bar=False, normalize_embeddings=True)
                for query, emb in zip(batch, embeddings):
                    query_embeddings_dict[query] = emb
            train_title_emb = embedder.encode(train_df['title_clean'].tolist(), batch_size=CONFIG['BATCH_SIZE'], show_progress_bar=True, normalize_embeddings=True)
            test_title_emb = embedder.encode(test_df['title_clean'].tolist(), batch_size=CONFIG['BATCH_SIZE'], show_progress_bar=True, normalize_embeddings=True)
            train_query_emb = np.array([query_embeddings_dict[q] for q in train_df['query_clean']])
            test_query_emb = np.array([query_embeddings_dict[q] for q in test_df['query_clean']])
            train_df['semantic_sim_bge'] = np.sum(train_query_emb * train_title_emb, axis=1)
            test_df['semantic_sim_bge'] = np.sum(test_query_emb * test_title_emb, axis=1)
            print("✓ BGE embeddings")
            del train_query_emb, test_query_emb, train_title_emb, test_title_emb
            gc.collect()
        except Exception as e:
            print(f" BGE error: {e}")
            train_df['semantic_sim_bge'] = 0.0
            test_df['semantic_sim_bge'] = 0.0
    else:
        train_df['semantic_sim_bge'] = 0.0
        test_df['semantic_sim_bge'] = 0.0
    # CROSS-ENCODER
    if CONFIG['USE_CROSS_ENCODER'] and SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            cross_encoder = CrossEncoder(CONFIG['CROSS_ENCODER_MODEL'])
            def compute_cross_encoder_scores(df, top_k=50):
                ce_scores = []
                for qid in tqdm(df['query_id'].unique(), desc="Cross-Encoder"):
                    mask = df['query_id'] == qid
                    qid_df = df[mask].copy()
                    query_text = qid_df['query_original'].iloc[0]
                    qid_df = qid_df.sort_values('bm25_okapi_title', ascending=False)
                    top_k_actual = min(top_k, len(qid_df))
                    pairs = [[query_text, title] for title in qid_df['title_original'].iloc[:top_k_actual]]
                    if pairs:
                        scores = cross_encoder.predict(pairs, show_progress_bar=False)
                        all_scores = np.full(len(qid_df), scores.min() - 1.0)
                        all_scores[:top_k_actual] = scores
                    else:
                        all_scores = np.zeros(len(qid_df))
                    ce_scores.extend(all_scores)
                return ce_scores
            train_df['cross_encoder_score'] = compute_cross_encoder_scores(train_df, CONFIG['CROSS_ENCODER_TOP_K'])
            test_df['cross_encoder_score'] = compute_cross_encoder_scores(test_df, CONFIG['CROSS_ENCODER_TOP_K'])
            print(" Cross-Encoder")
        except Exception as e:
            print(f" Cross-Encoder error: {e}")
            train_df['cross_encoder_score'] = 0.0
            test_df['cross_encoder_score'] = 0.0
    else:
        train_df['cross_encoder_score'] = 0.0
        test_df['cross_encoder_score'] = 0.0
    # PAIRWISE FEATURES
    pairwise_cols = ['bm25_okapi_title', 'semantic_sim_bge', 'cross_encoder_score', 'tfidf_cosine']
    train_pairwise = compute_pairwise_features(train_df, pairwise_cols)
    test_pairwise = compute_pairwise_features(test_df, pairwise_cols)
    # RRF
    rrf_cols = ['bm25_okapi_title', 'semantic_sim_bge', 'cross_encoder_score', 'tfidf_cosine']
    train_df['rrf_score'] = reciprocal_rank_fusion(train_df, rrf_cols, k=60)
    test_df['rrf_score'] = reciprocal_rank_fusion(test_df, rrf_cols, k=60)
    # ОБЪЕДИНЕНИЕ ПРИЗНАКОВ
    for df in [train_text_features, test_text_features, train_similarity, test_similarity,
               train_advanced, test_advanced, train_ngram, test_ngram, train_svd_df, test_svd_df,
               train_w2v_features, test_w2v_features, train_pairwise, test_pairwise,
               train_entity, test_entity, train_noise, test_noise]:
        df.reset_index(drop=True, inplace=True)
    train_features = pd.concat([train_text_features, train_similarity, train_advanced, train_ngram,
                                train_svd_df, train_w2v_features, train_pairwise, train_entity, train_noise], axis=1)
    test_features = pd.concat([test_text_features, test_similarity, test_advanced, test_ngram,
                               test_svd_df, test_w2v_features, test_pairwise, test_entity, test_noise], axis=1)
    additional_cols = ['bm25_okapi_title', 'bm25_plus_title', 'bm25_l_title', 'bm25_okapi_product',
                       'query_avg_title_len', 'query_std_title_len', 'query_item_count',
                       'semantic_sim_bge', 'cross_encoder_score', 'tfidf_cosine', 'rrf_score']
    if 'query_id_target_enc' in train_df.columns:
        additional_cols.append('query_id_target_enc')
    for col in additional_cols:
        if col in train_df.columns:
            train_features[col] = train_df[col].values
        if col in test_df.columns:
            test_features[col] = test_df[col].values
    # Комбинированные признаки
    train_features['bm25_x_match'] = train_features['bm25_okapi_title'] * train_features['title_match_ratio']
    train_features['semantic_x_match'] = train_features['semantic_sim_bge'] * train_features['title_match_ratio']
    train_features['semantic_x_bm25'] = train_features['semantic_sim_bge'] * train_features['bm25_okapi_title']
    train_features['cross_x_bm25'] = train_features['cross_encoder_score'] * train_features['bm25_okapi_title']
    test_features['bm25_x_match'] = test_features['bm25_okapi_title'] * test_features['title_match_ratio']
    test_features['semantic_x_match'] = test_features['semantic_sim_bge'] * test_features['title_match_ratio']
    test_features['semantic_x_bm25'] = test_features['semantic_sim_bge'] * test_features['bm25_okapi_title']
    test_features['cross_x_bm25'] = test_features['cross_encoder_score'] * test_features['bm25_okapi_title']
    # Interaction features
    if CONFIG['USE_INTERACTION_FEATURES']:
        for f1, f2 in [('brand_match', 'title_match_ratio'), ('color_match', 'title_match_ratio'), ('number_overlap', 'bm25_okapi_title')]:
            if f1 in train_features.columns and f2 in train_features.columns:
                train_features[f'{f1}_x_{f2}'] = train_features[f1] * train_features[f2]
                test_features[f'{f1}_x_{f2}'] = test_features[f1] * test_features[f2]
    # Полиномиальные
    for feat in ['bm25_okapi_title', 'semantic_sim_bge', 'cross_encoder_score', 'title_match_ratio']:
        if feat in train_features.columns:
            train_features[f'{feat}_sq'] = train_features[feat] ** 2
            test_features[f'{feat}_sq'] = test_features[feat] ** 2
    # Логарифмические
    for feat in ['query_len', 'title_len', 'query_item_count']:
        if feat in train_features.columns:
            train_features[f'{feat}_log'] = np.log1p(train_features[feat])
            test_features[f'{feat}_log'] = np.log1p(test_features[feat])
    train_features = train_features.fillna(0).replace([np.inf, -np.inf], 0)
    test_features = test_features.fillna(0).replace([np.inf, -np.inf], 0)
    train_features = optimize_dtypes(train_features)
    test_features = optimize_dtypes(test_features)
    print(f" Всего признаков: {train_features.shape[1]}")
    # FEATURE SELECTION
    if 'relevance' in train_df.columns and CONFIG['FEATURE_SELECTION_K'] < train_features.shape[1]:
        X_full = train_features.values.astype(np.float32)
        y_full = train_df['relevance'].values.astype(np.float32)
        print(f"Selecting top {CONFIG['FEATURE_SELECTION_K']} from {train_features.shape[1]}...")
        selector = SelectKBest(mutual_info_regression, k=CONFIG['FEATURE_SELECTION_K'])
        selector.fit(X_full, y_full)
        selected_indices = selector.get_support(indices=True)
        selected_features = [train_features.columns[i] for i in selected_indices]
        print("Top 20:", selected_features[:20])
        train_features = train_features.iloc[:, selected_indices]
        test_features = test_features.iloc[:, selected_indices]
        print(f"After selection: {train_features.shape[1]} features")
    # VALIDATION SPLIT
    if CONFIG['USE_VALIDATION'] and 'relevance' in train_df.columns:
        unique_queries = train_df['query_id'].unique()
        np.random.shuffle(unique_queries)
        val_size = int(len(unique_queries) * CONFIG['VALIDATION_SIZE'])
        val_queries = set(unique_queries[:val_size])
        val_mask = train_df['query_id'].isin(val_queries)
        X_train_split = train_features[~val_mask].values.astype(np.float32)
        y_train_split = train_df[~val_mask]['relevance'].values.astype(np.float32)
        query_train_split = train_df[~val_mask]['query_id'].values
        X_val_split = train_features[val_mask].values.astype(np.float32)
        y_val_split = train_df[val_mask]['relevance'].values.astype(np.float32)
        query_val_split = train_df[val_mask]['query_id'].values
        train_groups_split = [len(group) for _, group in pd.Series(query_train_split).groupby(query_train_split)]
        val_groups_split = [len(group) for _, group in pd.Series(query_val_split).groupby(query_val_split)]
        print(f"Train: {len(X_train_split)}, Val: {len(X_val_split)}")
    else:
        X_train_split = train_features.values.astype(np.float32)
        y_train_split = train_df['relevance'].values.astype(np.float32)
        query_train_split = train_df['query_id'].values
        train_groups_split = [len(group) for _, group in pd.Series(query_train_split).groupby(query_train_split)]
        X_val_split = None
        y_val_split = None
        val_groups_split = None
    # LIGHTGBM
    if 'relevance' in train_df.columns:
        train_data = lgb.Dataset(X_train_split, label=y_train_split, group=train_groups_split)
        val_data = lgb.Dataset(X_val_split, label=y_val_split, group=val_groups_split) if X_val_split is not None else None
        params_lgb = {
            'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [10],
            'learning_rate': 0.02, 'num_leaves': 127, 'min_data_in_leaf': 10,
            'feature_fraction': 0.75, 'bagging_fraction': 0.75, 'bagging_freq': 5,
            'lambda_l1': 0.05, 'lambda_l2': 0.05, 'max_depth': 10,
            'verbosity': -1, 'n_jobs': CONFIG['N_JOBS'], 'seed': CONFIG['SEED']
        }
        callbacks = [lgb.log_evaluation(50)]
        if val_data is not None:
            callbacks.append(lgb.early_stopping(75))
        model_lgb = lgb.train(params_lgb, train_data, num_boost_round=3000,
                              valid_sets=[val_data] if val_data is not None else None, callbacks=callbacks)
        print(f"\n✓ LightGBM trained!")
    # CATBOOST
    CATBOOST_TRAINED = False
    if CONFIG['USE_CATBOOST'] and CATBOOST_AVAILABLE and 'relevance' in train_df.columns:
        try:
            train_pool = Pool(X_train_split, label=y_train_split, group_id=query_train_split)
            val_pool = Pool(X_val_split, label=y_val_split, group_id=query_val_split) if X_val_split is not None else None
            model_cat = CatBoostRanker(iterations=1500, learning_rate=0.02, depth=10, loss_function='YetiRank',
                                       custom_metric=['NDCG:top=10'], random_seed=CONFIG['SEED'], verbose=100,
                                       use_best_model=True if val_pool else False, od_type='Iter' if val_pool else None,
                                       od_wait=75 if val_pool else None)
            model_cat.fit(train_pool, eval_set=val_pool, verbose=100, plot=False)
            print(f"\n CatBoost trained!")
            CATBOOST_TRAINED = True
        except Exception as e:
            print(f"CatBoost error: {e}")
    # XGBOOST
    XGBOOST_TRAINED = False
    if CONFIG['USE_XGBOOST'] and XGBOOST_AVAILABLE and 'relevance' in train_df.columns:
        try:
            dtrain = xgb.DMatrix(X_train_split, label=y_train_split)
            dtrain.set_group(train_groups_split)
            if X_val_split is not None:
                dval = xgb.DMatrix(X_val_split, label=y_val_split)
                dval.set_group(val_groups_split)
            params_xgb = {'objective': 'rank:ndcg', 'eta': 0.03, 'max_depth': 10, 'subsample': 0.75,
                          'colsample_bytree': 0.75, 'eval_metric': 'ndcg@10', 'seed': CONFIG['SEED'], 'tree_method': 'hist'}
            model_xgb = xgb.train(params_xgb, dtrain, num_boost_round=1500,
                                  evals=[(dval, 'val')] if X_val_split is not None else None,
                                  early_stopping_rounds=75, verbose_eval=50)
            print(f"\n XGBoost trained!")
            XGBOOST_TRAINED = True
        except Exception as e:
            print(f"XGBoost error: {e}")
    # ENSEMBLE
    if 'relevance' in train_df.columns:
        X_test_final = test_features.values.astype(np.float32)
        all_predictions = []
        weights = []
        pred_lgb = model_lgb.predict(X_test_final)
        all_predictions.append(pred_lgb)
        weights.append(0.40)
        print(f"LightGBM: [{pred_lgb.min():.4f}, {pred_lgb.max():.4f}]")
        if CATBOOST_TRAINED:
            pred_cat = model_cat.predict(X_test_final)
            all_predictions.append(pred_cat)
            weights.append(0.35)
            print(f"CatBoost: [{pred_cat.min():.4f}, {pred_cat.max():.4f}]")
        if XGBOOST_TRAINED:
            dtest = xgb.DMatrix(X_test_final)
            pred_xgb = model_xgb.predict(dtest)
            all_predictions.append(pred_xgb)
            weights.append(0.20)
            print(f"XGBoost: [{pred_xgb.min():.4f}, {pred_xgb.max():.4f}]")
        pred_rrf = test_df['rrf_score'].values
        all_predictions.append(pred_rrf)
        weights.append(0.05)
        print(f"RRF: [{pred_rrf.min():.4f}, {pred_rrf.max():.4f}]")
        weights = np.array(weights)
        weights = weights / weights.sum()
        model_names = ['LGB', 'CAT', 'XGB', 'RRF'][:len(weights)]
        print(f"\nEnsemble weights: {dict(zip(model_names, weights))}")
        predictions_raw = np.zeros(len(X_test_final))
        for pred, weight in zip(all_predictions, weights):
            predictions_raw += weight * pred
        print(f"Raw ensemble: [{predictions_raw.min():.4f}, {predictions_raw.max():.4f}]")
        # POST-PROCESSING
        query_test = test_df['query_id'].values
        predictions_calibrated = calibrate_predictions(predictions_raw, query_test, method='minmax')
        print(f"After calibration: [{predictions_calibrated.min():.4f}, {predictions_calibrated.max():.4f}]")
        predictions_final = 0.8 * predictions_raw + 0.2 * predictions_calibrated
        print(f"Final blend: [{predictions_final.min():.4f}, {predictions_final.max():.4f}]")
        # Создание submission файла
        create_submission(predictions_final, test_ids_original.values)
if __name__ == "__main__":
    main()