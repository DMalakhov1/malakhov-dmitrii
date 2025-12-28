"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""

import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from scipy.optimize import minimize


def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """
    
    # Создать пандас таблицу submission
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    
    predictions.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    print("\nПервые 5 строк submission:")
    print(predictions.head())
    
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
    
    np.random.seed(322)
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    sample = pd.read_csv('data/sample_submission.csv')

    # УМНАЯ ОБРАБОТКА ВЫБРОСОВ
    def remove_outliers_percentile(df, columns, lower_percentile=3, upper_percentile=97):
        df_clean = df.copy()
        initial_len = len(df_clean)
        for col in columns:
            lower_bound = df_clean[col].quantile(lower_percentile / 100)
            upper_bound = df_clean[col].quantile(upper_percentile / 100)
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            df_clean = df_clean[mask]
        total_removed = initial_len - len(df_clean)
        print(f"Удалено {total_removed} выбросов ({total_removed/initial_len*100:.2f}%)")
        return df_clean

    # Очистка целевых переменных
    target_cols = ['price_p05', 'price_p95']
    train_clean = remove_outliers_percentile(train, target_cols, lower_percentile=3, upper_percentile=97)

    def winsorize_features(df, columns, limits=(0.01, 0.99)):
        df_winsorized = df.copy()
        for col in columns:
            lower = df[col].quantile(limits[0])
            upper = df[col].quantile(limits[1])
            df_winsorized[col] = df_winsorized[col].clip(lower, upper)
        return df_winsorized

    numeric_cols = ['n_stores', 'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level']
    train_clean = winsorize_features(train_clean, numeric_cols, limits=(0.01, 0.99))
    test = winsorize_features(test, numeric_cols, limits=(0.01, 0.99))

    # FEATURE ENGINEERING
    def advanced_feature_engineering(df):
        df = df.copy()
        df['dt'] = pd.to_datetime(df['dt'])
        df['year'] = df['dt'].dt.year
        df['quarter'] = df['dt'].dt.quarter
        df['day_of_year'] = df['dt'].dt.dayofyear
        df['week_of_year_norm'] = df['week_of_year'] / 52
        df['is_weekend'] = (df['dow'] >= 5).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 7).astype(int)
        df['is_month_middle'] = ((df['day_of_month'] > 7) & (df['day_of_month'] <= 21)).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 22).astype(int)
        df['is_quarter_start'] = df['dt'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['dt'].dt.is_quarter_end.astype(int)
        
        # ЦИКЛИЧЕСКИЕ ПРИЗНАКИ 
        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        # ВЗАИМОДЕЙСТВИЯ ПОГОДНЫХ ПРИЗНАКОВ 
        df['temp_humidity'] = df['avg_temperature'] * df['avg_humidity']
        df['temp_wind'] = df['avg_temperature'] * df['avg_wind_level']
        df['humidity_wind'] = df['avg_humidity'] * df['avg_wind_level']
        df['temp_precpt'] = df['avg_temperature'] * df['precpt']
        df['humidity_precpt'] = df['avg_humidity'] * df['precpt']
        df['wind_precpt'] = df['avg_wind_level'] * df['precpt']
        
        # Трехсторонние взаимодействия
        df['temp_humidity_wind'] = df['avg_temperature'] * df['avg_humidity'] * df['avg_wind_level']
        df['temp_humidity_precpt'] = df['avg_temperature'] * df['avg_humidity'] * df['precpt']
        
        # ПОЛИНОМИАЛЬНЫЕ ПРИЗНАКИ 
        df['temp_squared'] = df['avg_temperature'] ** 2
        df['temp_cubed'] = df['avg_temperature'] ** 3
        df['humidity_squared'] = df['avg_humidity'] ** 2
        df['humidity_cubed'] = df['avg_humidity'] ** 3
        df['wind_squared'] = df['avg_wind_level'] ** 2
        df['precpt_squared'] = df['precpt'] ** 2
        df['n_stores_squared'] = df['n_stores'] ** 2 
        
        # Корневые признаки
        df['temp_sqrt'] = np.sqrt(np.abs(df['avg_temperature']))
        df['humidity_sqrt'] = np.sqrt(np.abs(df['avg_humidity']))
        
        # КОМБИНАЦИИ ФЛАГОВ 
        df['holiday_activity'] = df['holiday_flag'] * df['activity_flag']
        df['holiday_weekend'] = df['holiday_flag'] * df['is_weekend']
        df['activity_weekend'] = df['activity_flag'] * df['is_weekend']
        df['holiday_activity_weekend'] = df['holiday_flag'] * df['activity_flag'] * df['is_weekend']
        
        # Флаги с погодой
        df['holiday_temp'] = df['holiday_flag'] * df['avg_temperature']
        df['activity_temp'] = df['activity_flag'] * df['avg_temperature']
        df['weekend_temp'] = df['is_weekend'] * df['avg_temperature']
        
        # ЛОГАРИФМИЧЕСКИЕ ПРЕОБРАЗОВАНИЯ 
        df['log_n_stores'] = np.log1p(df['n_stores'] + abs(df['n_stores'].min()) + 1)
        df['log_precpt'] = np.log1p(df['precpt'] + abs(df['precpt'].min()) + 1)
        
        # ЧАСТОТНОЕ КОДИРОВАНИЕ 
        cat_cols = ['product_id', 'first_category_id', 'second_category_id',
                    'third_category_id', 'management_group_id']
        for cat_col in cat_cols:
            freq = df[cat_col].value_counts(normalize=True)
            df[f'{cat_col}_freq'] = df[cat_col].map(freq).fillna(0)
            count = df[cat_col].value_counts()
            df[f'{cat_col}_count'] = df[cat_col].map(count).fillna(0)
            df[f'{cat_col}_log_count'] = np.log1p(df[f'{cat_col}_count'])
            
        # СТАТИСТИКА ПО ГРУППАМ 
        
        # По product_id
        df['product_mean_temp'] = df.groupby('product_id')['avg_temperature'].transform('mean')
        df['product_std_temp'] = df.groupby('product_id')['avg_temperature'].transform('std').fillna(0)
        df['product_mean_humidity'] = df.groupby('product_id')['avg_humidity'].transform('mean')
        df['product_mean_n_stores'] = df.groupby('product_id')['n_stores'].transform('mean')
        
        # Отклонение от среднего
        df['temp_diff_from_product_mean'] = df['avg_temperature'] - df['product_mean_temp']
        df['humidity_diff_from_product_mean'] = df['avg_humidity'] - df['product_mean_humidity']
        df['n_stores_diff_from_product_mean'] = df['n_stores'] - df['product_mean_n_stores']
        
        # По категориям
        for cat in ['first_category_id', 'second_category_id', 'third_category_id']:
            df[f'{cat}_mean_temp'] = df.groupby(cat)['avg_temperature'].transform('mean')
            df[f'{cat}_mean_n_stores'] = df.groupby(cat)['n_stores'].transform('mean')
            
        # РАНКОВЫЕ ПРИЗНАКИ 
        df['temp_rank'] = df.groupby('product_id')['avg_temperature'].rank(pct=True)
        df['humidity_rank'] = df.groupby('product_id')['avg_humidity'].rank(pct=True)
        df['n_stores_rank'] = df.groupby('product_id')['n_stores'].rank(pct=True)
        
        # КОМФОРТНЫЙ ИНДЕКС (погода)
        df['comfort_index'] = (
            0.4 * (1 - np.abs(df['avg_temperature'])) +
            0.3 * (1 - np.abs(df['avg_humidity'])) +
            0.2 * (1 - np.abs(df['avg_wind_level'])) +
            0.1 * (1 - np.abs(df['precpt']))
        )
        # СЕЗОННОСТЬ 
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        return df

    train_fe = advanced_feature_engineering(train_clean)
    test_fe = advanced_feature_engineering(test)
    print(f"Создано {train_fe.shape[1] - train.shape[1]} новых признаков")

    # LAG FEATURES + ROLLING STATISTICS
    def create_advanced_lag_features(df, is_train=True):
        df = df.copy()
        df = df.sort_values(['product_id', 'dt'])
        # ЛАГИ ДЛЯ ЦЕН (только для train) 
        if is_train:
            for lag in [1, 3, 7, 14, 21]:
                df[f'price_p05_lag_{lag}'] = df.groupby('product_id')['price_p05'].shift(lag)
                df[f'price_p95_lag_{lag}'] = df.groupby('product_id')['price_p95'].shift(lag)
                df[f'price_spread_lag_{lag}'] = df[f'price_p95_lag_{lag}'] - df[f'price_p05_lag_{lag}']
                
        # ROLLING STATISTICS 
        windows = [3, 7, 14]
        features_to_roll = ['n_stores', 'avg_temperature', 'avg_humidity', 'precpt', 'avg_wind_level']
        for window in windows:
            for feature in features_to_roll:
                # Mean
                df[f'{feature}_rolling_mean_{window}'] = df.groupby('product_id')[feature].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                # Std
                df[f'{feature}_rolling_std_{window}'] = df.groupby('product_id')[feature].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                ).fillna(0)
                # Min/Max
                df[f'{feature}_rolling_min_{window}'] = df.groupby('product_id')[feature].transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
                df[f'{feature}_rolling_max_{window}'] = df.groupby('product_id')[feature].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
                # Range
                df[f'{feature}_rolling_range_{window}'] = (
                    df[f'{feature}_rolling_max_{window}'] - df[f'{feature}_rolling_min_{window}']
                )
        # EXPANDING STATISTICS 
        for feature in features_to_roll:
            df[f'{feature}_expanding_mean'] = df.groupby('product_id')[feature].transform(
                lambda x: x.expanding(min_periods=1).mean()
            )
            df[f'{feature}_expanding_std'] = df.groupby('product_id')[feature].transform(
                lambda x: x.expanding(min_periods=1).std()
            ).fillna(0)
        # РАЗНИЦА И СКОРОСТЬ ИЗМЕНЕНИЯ
        for feature in features_to_roll:
            df[f'{feature}_diff'] = df.groupby('product_id')[feature].diff().fillna(0)
            df[f'{feature}_pct_change'] = df.groupby('product_id')[feature].pct_change().fillna(0)
            # Ускорение (вторая производная)
            df[f'{feature}_diff_2'] = df.groupby('product_id')[f'{feature}_diff'].diff().fillna(0)
        # TREND FEATURES 
        for feature in features_to_roll:
            # Trend за последние 7 дней
            df[f'{feature}_trend_7'] = df.groupby('product_id')[feature].transform(
                lambda x: x.rolling(window=7, min_periods=2).apply(
                    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0, raw=True
                )
            ).fillna(0)
        return df

    train_fe = create_advanced_lag_features(train_fe, is_train=True)
    test_fe = create_advanced_lag_features(test_fe, is_train=False)
    print(f"✓ Lag features созданы, всего признаков: {train_fe.shape[1]}")

    # TARGET ENCODING (улучшенное)
    def target_encoding_advanced(train, test, cat_cols, target_col, smoothing=10):
        train_encoded = train.copy()
        test_encoded = test.copy()
        for col in cat_cols:
            # Глобальное среднее
            global_mean = train[target_col].mean()
            # Агрегация
            agg = train.groupby(col)[target_col].agg(['mean', 'count', 'std']).reset_index()
            agg.columns = [col, 'mean', 'count', 'std']
            # Сглаживание
            smoothing_factor = 1 / (1 + np.exp(-(agg['count'] - smoothing) / smoothing))
            agg['smoothed_mean'] = global_mean * (1 - smoothing_factor) + agg['mean'] * smoothing_factor
            agg['smoothed_std'] = agg['std'].fillna(0)
            # Маппинг
            mean_map = dict(zip(agg[col], agg['smoothed_mean']))
            std_map = dict(zip(agg[col], agg['smoothed_std']))
            train_encoded[f'{col}_target_mean_{target_col}'] = train[col].map(mean_map).fillna(global_mean)
            test_encoded[f'{col}_target_mean_{target_col}'] = test[col].map(mean_map).fillna(global_mean)
            train_encoded[f'{col}_target_std_{target_col}'] = train[col].map(std_map).fillna(0)
            test_encoded[f'{col}_target_std_{target_col}'] = test[col].map(std_map).fillna(0)
        return train_encoded, test_encoded

    cat_cols_for_encoding = ['product_id', 'first_category_id', 'second_category_id',
                             'third_category_id', 'management_group_id']
    train_fe, test_fe = target_encoding_advanced(train_fe, test_fe, cat_cols_for_encoding, 'price_p05', smoothing=10)
    train_fe, test_fe = target_encoding_advanced(train_fe, test_fe, cat_cols_for_encoding, 'price_p95', smoothing=10)
    print(f"✓ Target encoding применен")

    # КЛАСТЕРИЗАЦИЯ (чисто для себя)
    # Кластеризация товаров
    product_features = train_fe.groupby('product_id').agg({
        'price_p05': ['mean', 'std', 'min', 'max'],
        'price_p95': ['mean', 'std', 'min', 'max'],
        'n_stores': ['mean', 'std'],
        'holiday_flag': 'sum',
        'activity_flag': 'sum',
        'avg_temperature': ['mean', 'std'],
        'avg_humidity': ['mean', 'std']
    }).reset_index()
    product_features.columns = ['_'.join(col).strip('_') for col in product_features.columns]

    scaler_cluster = RobustScaler()
    cluster_cols = [col for col in product_features.columns if col != 'product_id']
    product_features_scaled = scaler_cluster.fit_transform(product_features[cluster_cols].fillna(0))

    # Product кластеры
    n_clusters_product = 20
    kmeans_product = KMeans(n_clusters=n_clusters_product, random_state=322, n_init=20)
    product_features['cluster_product'] = kmeans_product.fit_predict(product_features_scaled)

    train_fe = train_fe.merge(product_features[['product_id', 'cluster_product']], on='product_id', how='left')
    test_fe = test_fe.merge(product_features[['product_id', 'cluster_product']], on='product_id', how='left')

    # Кластеризация по категориям
    category_features = train_fe.groupby('first_category_id').agg({
        'price_p05': 'mean',
        'price_p95': 'mean',
        'n_stores': 'mean'
    }).reset_index()

    scaler_cat = RobustScaler()
    cat_features_scaled = scaler_cat.fit_transform(category_features[['price_p05', 'price_p95', 'n_stores']])

    n_clusters_cat = 10
    kmeans_cat = KMeans(n_clusters=n_clusters_cat, random_state=322, n_init=10)
    category_features['cluster_category'] = kmeans_cat.fit_predict(cat_features_scaled)

    train_fe = train_fe.merge(category_features[['first_category_id', 'cluster_category']], on='first_category_id', how='left')
    test_fe = test_fe.merge(category_features[['first_category_id', 'cluster_category']], on='first_category_id', how='left')

    print(f"Product кластеров: {n_clusters_product}, Category кластеров: {n_clusters_cat}")

    # ДЕТЕКЦИЯ АНОМАЛИЙ
    iso_forest = IsolationForest(contamination=0.02, random_state=322, n_jobs=-1)
    numeric_for_anomaly = ['n_stores', 'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level']
    train_fe['anomaly_score'] = iso_forest.fit_predict(train_fe[numeric_for_anomaly].fillna(0))
    test_fe['anomaly_score'] = iso_forest.predict(test_fe[numeric_for_anomaly].fillna(0))

    # Преобразуем в бинарный признак
    train_fe['is_anomaly'] = (train_fe['anomaly_score'] == -1).astype(int)
    test_fe['is_anomaly'] = (test_fe['anomaly_score'] == -1).astype(int)

    anomalies = train_fe['is_anomaly'].sum()
    print(f" Обнаружено аномалий: {anomalies} ({anomalies/len(train_fe)*100:.2f}%)")

    # PCA ДЛЯ ПОГОДНЫХ ПРИЗНАКОВ
    weather_features = ['avg_temperature', 'avg_humidity', 'avg_wind_level', 'precpt']
    scaler_pca = StandardScaler()
    train_weather_scaled = scaler_pca.fit_transform(train_fe[weather_features])
    test_weather_scaled = scaler_pca.transform(test_fe[weather_features])

    pca = PCA(n_components=3, random_state=322)
    train_pca = pca.fit_transform(train_weather_scaled)
    test_pca = pca.transform(test_weather_scaled)

    for i in range(3):
        train_fe[f'weather_pca_{i}'] = train_pca[:, i]
        test_fe[f'weather_pca_{i}'] = test_pca[:, i]

    print(f" PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # ПОДГОТОВКА ДАННЫХ
    test_row_ids = test_fe['row_id'].copy()
    y_p05 = train_fe['price_p05'].values
    y_p95 = train_fe['price_p95'].values

    drop_cols = ['dt', 'price_p05', 'price_p95', 'row_id']
    drop_cols_train = [col for col in drop_cols if col in train_fe.columns]
    drop_cols_test = [col for col in drop_cols if col in test_fe.columns]

    X_train = train_fe.drop(columns=drop_cols_train)
    X_test = test_fe.drop(columns=drop_cols_test)

    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    # Заполнение пропусков
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())

    # Заменяем inf на большие числа
    X_train = X_train.replace([np.inf, -np.inf], [1e10, -1e10])
    X_test = X_test.replace([np.inf, -np.inf], [1e10, -1e10])

    # ОБУЧЕНИЕ МОДЕЛЕЙ
    def iou_score(y_true_p05, y_true_p95, y_pred_p05, y_pred_p95, epsilon=1e-6):
        ious = []
        for lt, ut, lp, up in zip(y_true_p05, y_true_p95, y_pred_p05, y_pred_p95):
            lt_adj = lt - epsilon / 2
            ut_adj = ut + epsilon / 2
            lp_adj = lp - epsilon / 2
            up_adj = up + epsilon / 2
            intersection = max(0, min(ut_adj, up_adj) - max(lt_adj, lp_adj))
            union = (ut_adj - lt_adj) + (up_adj - lp_adj) - intersection
            iou = intersection / union if union > 0 else 0
            ious.append(iou)
        return np.mean(ious)

    def optimize_weights(preds_p05, preds_p95, y_val_p05, y_val_p95):
        """Оптимизация весов ансамбля по IoU"""
        def objective(weights):
            w_lgb, w_gb, w_rf, w_qr = weights[:4]
            pred_p05 = w_lgb * preds_p05['lgb'] + w_gb * preds_p05['gb'] + w_rf * preds_p05['rf'] + w_qr * preds_p05['qr']
            w_lgb, w_gb, w_rf, w_qr = weights[4:]
            pred_p95 = w_lgb * preds_p95['lgb'] + w_gb * preds_p95['gb'] + w_rf * preds_p95['rf'] + w_qr * preds_p95['qr']
            return -iou_score(y_val_p05, y_val_p95, pred_p05, pred_p95)
        
        initial_weights = [0.5, 0.25, 0.15, 0.1] * 2
        bounds = [(0, 1)] * 8
        constraints = [{'type': 'eq', 'fun': lambda w: sum(w[:4]) - 1},
                       {'type': 'eq', 'fun': lambda w: sum(w[4:]) - 1}]
        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x[:4], result.x[4:]

    def train_optimized_ensemble(X_train_full, y_train, X_test, target_name='price', quantile=0.5):
        """Оптимизированный ансамбль"""
        print(f"\n>>> Обучение для {target_name}")
        val_size = int(0.2 * len(X_train_full))
        X_tr = X_train_full.iloc[: -val_size]
        X_val = X_train_full.iloc[-val_size :]
        y_tr = y_train[:-val_size]
        y_val = y_train[-val_size:]
        
        # LIGHTGBM
        print("LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=quantile,
            n_estimators=2000,
            learning_rate=0.02,
            max_depth=4,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=322,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        print(f" Best iteration: {lgb_model.best_iteration_}")
        
        # GRADIENT BOOSTING
        print("Gradient Boosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=20,
            min_samples_leaf=15,
            subsample=0.8,
            max_features='sqrt',
            random_state=322,
            verbose=0
        )
        gb_model.fit(X_tr, y_tr)
        
        # RANDOM FOREST
        print("Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            random_state=322,
            verbose=0
        )
        rf_model.fit(X_tr, y_tr)
        
        # QUANTILE REGRESSION
        print("Quantile Regression...")
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        qr_model = QuantileRegressor(
            quantile=quantile,
            alpha=0.1,
            solver='highs'
        )
        qr_model.fit(X_tr_scaled, y_tr)
        
        # Предсказания на val и test
        pred_val = {
            'lgb': lgb_model.predict(X_val),
            'gb': gb_model.predict(X_val),
            'rf': rf_model.predict(X_val),
            'qr': qr_model.predict(X_val_scaled)
        }
        pred_test = {
            'lgb': lgb_model.predict(X_test),
            'gb': gb_model.predict(X_test),
            'rf': rf_model.predict(X_test),
            'qr': qr_model.predict(X_test_scaled)
        }
        return pred_val, pred_test, lgb_model

    # Обучаем модели для p05 и p95
    print("\nМодель для price_p05")
    pred_val_p05, pred_test_p05, model_p05 = train_optimized_ensemble(X_train, y_p05, X_test, 'price_p05', quantile=0.05)

    print("\nМодель для price_p95")
    pred_val_p95, pred_test_p95, model_p95 = train_optimized_ensemble(X_train, y_p95, X_test, 'price_p95', quantile=0.95)

    # Оптимизация весов
    val_size = int(0.2 * len(X_train))
    y_val_p05 = y_p05[-val_size:]
    y_val_p95 = y_p95[-val_size:]

    weights_p05, weights_p95 = optimize_weights(pred_val_p05, pred_val_p95, y_val_p05, y_val_p95)
    print(f"Оптимизированные веса p05: {weights_p05}")
    print(f"Оптимизированные веса p95: {weights_p95}")

    # Финальные предсказания на test
    final_pred_p05 = (
        weights_p05[0] * pred_test_p05['lgb'] +
        weights_p05[1] * pred_test_p05['gb'] +
        weights_p05[2] * pred_test_p05['rf'] +
        weights_p05[3] * pred_test_p05['qr'])

    final_pred_p95 = (
        weights_p95[0] * pred_test_p95['lgb'] +
        weights_p95[1] * pred_test_p95['gb'] +
        weights_p95[2] * pred_test_p95['rf'] +
        weights_p95[3] * pred_test_p95['qr'])

    # POST-PROCESSING
    epsilon = 1e-6
    pred_p05_adjusted = np.minimum(final_pred_p05, final_pred_p95 - epsilon)
    pred_p95_adjusted = np.maximum(final_pred_p95, pred_p05_adjusted + epsilon)

    print(f"Среднее p05: {pred_p05_adjusted.mean():.4f}")
    print(f"Среднее p95: {pred_p95_adjusted.mean():.4f}")
    print(f"Средняя ширина интервала: {(pred_p95_adjusted - pred_p05_adjusted).mean():.4f}")

    # SUBMISSION
    predictions = pd.DataFrame({
        'row_id': test_row_ids,
        'price_p05': pred_p05_adjusted,
        'price_p95': pred_p95_adjusted
    })
    
    # ВАЖНОСТЬ ПРИЗНАКОВ
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model_p05.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nТоп-30 важных признаков:")
    print(feature_importance.head(30).to_string(index=False))
    
    # ФИНАЛ
    print(f"\nПризнаков: {X_train.shape[1]}")
    print(f"Обучающих примеров: {len(X_train)}")
    print(f"Предсказаний: {len(predictions)}")
    
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(predictions)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)
    
    return predictions


if __name__ == "__main__":
    main()