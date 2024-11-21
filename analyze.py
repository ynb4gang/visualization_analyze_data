from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from autots import AutoTS
import matplotlib.pyplot as plt
import pandas as pd


def analyze_regression(file_path, target_column):
    print("=== Линейная регрессия ===")
    data = pd.read_csv(file_path)

    if target_column not in data.columns:
        raise ValueError(f"Столбец '{target_column}' отсутствует в данных.")

    X = data.drop(columns=[target_column, 'customer_id'])
    y = data[target_column]

    numeric_features = ['product_views', 'ad_clicks']
    categorical_features = ['region', 'customer_segment']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2))
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = RandomForestRegressor(random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model)])

    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X, y)


    best_model = grid_search.best_estimator_
    data['Predictions'] = best_model.predict(X)

    mae = mean_absolute_error(y, data['Predictions'])
    rmse = mean_squared_error(y, data['Predictions'], squared=False)
    r2 = r2_score(y, data['Predictions'])
    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"MAE: {mae}, RMSE: {rmse}, R²: {r2}")


    plt.figure(figsize=(10, 6))
    plt.scatter(y, data['Predictions'], alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()],
             color='red', linestyle='--', linewidth=2, label='Идеальная линия')
    plt.title('Фактические значения vs Прогнозы')
    plt.xlabel('Фактические значения')
    plt.ylabel('Прогнозы')
    plt.legend()
    plt.grid(True)
    plot_path = "regression_plot_improved.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"График регрессии сохранен в '{plot_path}'.")
    plt.show()

    output_path = "regression_predictions_improved.csv"
    data.to_csv(output_path, index=False)
    print(f"Прогнозы линейной регрессии сохранены в '{output_path}'.")


def analyze_clustering(file_path, features):
    print("=== Кластеризация ===")
    data = pd.read_csv(file_path)

    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        raise ValueError(f"Следующие столбцы отсутствуют в данных: {missing_features}")

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    silhouette_scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_scaled)
        score = silhouette_score(data_scaled, kmeans.labels_)
        silhouette_scores.append(score)
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    print(f"Оптимальное число кластеров: {optimal_k}")

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data_scaled)

    silhouette_avg = silhouette_score(data_scaled, data['Cluster'])
    print(f"Silhouette Score: {silhouette_avg}")

    plt.figure(figsize=(8, 6))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=data['Cluster'], cmap='viridis')
    plt.title('Кластеры после PCA')
    plt.xlabel('Первая главная компонента')
    plt.ylabel('Вторая главная компонента')
    plt.colorbar(label='Кластеры')
    plt.grid(True)
    plot_path = "clustering_plot_improved.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"График кластеризации сохранен в '{plot_path}'.")
    plt.show()

    output_path = "clusters_results_improved.csv"
    data.to_csv(output_path, index=False)
    print(f"Результаты кластеризации сохранены в '{output_path}'.")


def analyze_timeseries(file_path, target_column):
    print("=== Анализ временных рядов ===")
    data = pd.read_csv(file_path)

    if target_column not in data.columns:
        raise ValueError(f"Столбец '{target_column}' отсутствует в данных.")
    if 'period_date' not in data.columns:
        raise ValueError("Для временных рядов требуется колонка 'period_date' с временным индексом.")

    data['period_date'] = pd.to_datetime(data['period_date'])
    data.sort_values('period_date', inplace=True)

    model = AutoTS(
        forecast_length=6,
        frequency='infer',
        prediction_interval=0.95,
        ensemble='simple',
        model_list="superfast"
    )
    model = model.fit(data, date_col='period_date', value_col=target_column)

    forecast = model.predict()
    forecast_df = forecast.forecast

    actuals = data[target_column][-len(forecast_df):]
    mae = mean_absolute_error(actuals, forecast_df)
    rmse = mean_squared_error(actuals, forecast_df, squared=False)
    print(f"MAE: {mae}, RMSE: {rmse}")

    plt.figure(figsize=(10, 6))
    plt.plot(data['period_date'], data[target_column], label='Исторические данные', marker='o')
    forecast_dates = pd.date_range(
        start=data['period_date'].iloc[-1],
        periods=len(forecast_df) + 1,
        freq='M'
    )[1:]
    plt.plot(forecast_dates, forecast_df, label='Прогноз', linestyle='--', marker='x')
    plt.title('Прогноз временных рядов')
    plt.xlabel('Дата')
    plt.ylabel(target_column)
    plt.legend()
    plt.grid(True)
    plot_path = "forecast_plot_improved.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"График прогноза сохранен в '{plot_path}'.")
    plt.show()

    output_path = "forecast_results_improved.csv"
    forecast_df.to_csv(output_path, index=False)
    print(f"Прогноз сохранен в '{output_path}'.")


if __name__ == "__main__":
    file_path = "data.csv"
    target_column = "monthly_spending"

    analyze_regression(file_path, target_column)
    analyze_clustering(file_path, ['product_views', 'ad_clicks', 'monthly_spending'])
    analyze_timeseries(file_path, target_column)
