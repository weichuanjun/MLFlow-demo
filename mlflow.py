import hydra
import mlflow
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import importlib
import pandas as pd
from sklearn.preprocessing import StandardScaler

def train_models(cfg, X, y):
    """训练多个模型并返回最佳参数和模型。"""
    best_estimators = {}
    
    for model_name, model_cfg in cfg.models.items():
        module = importlib.import_module(model_cfg.module)
        model_class = getattr(module, model_cfg.model_class)
        model = model_class()
        param_grid = model_cfg.hyperparameters
        
        grid = GridSearchCV(model, param_grid, cv=cfg.GridSearchCV.cv, scoring=cfg.GridSearchCV.scoring, n_jobs=cfg.GridSearchCV.n_jobs)
        grid.fit(X, y)
        best_estimators[model_name] = grid.best_estimator_
        
        # 在训练时记录MLflow日志
        with mlflow.start_run(run_name=f"{model_name}_training", experiment_id=cfg.mlflow.experiment_id):
            mlflow.log_params(grid.best_params_)
            mlflow.sklearn.log_model(grid.best_estimator_, model_name)

    return best_estimators

def evaluate_models(cfg, models, X_test, y_test):
    results = {}  
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name}_evaluation", experiment_id=cfg.mlflow.experiment_id, nested=True):
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[model_name] = {'MSE': mse, 'R2': r2}
            mlflow.log_metric('MSE', mse)
            mlflow.log_metric('R2', r2)
            mlflow.sklearn.log_model(model, artifact_path="model")
    
    return results

@hydra.main(config_path="configs", config_name="parameters",version_base=None)
def main(config):
    # 加载数据集（示例数据集）
    data = pd.read_csv('dataset/Boston Housing.csv')

    X = data.drop('medv', axis=1)
    y = data['medv']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练模型
    best_estimators = train_models(config, X_train_scaled, y_train)
    
    # 评估模型
    results = evaluate_models(config, best_estimators, X_test_scaled, y_test)
    print(results)

if __name__ == "__main__":
    main()