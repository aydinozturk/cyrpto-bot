"""
Makine öğrenmesi modellerini eğiten ve değerlendiren modül.
Bagged Tree, Random Forest, Karar Ağacı ve diğer modelleri içerir.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class CryptoMLModels:
    def __init__(self):
        """Makine öğrenmesi modelleri sınıfını başlatır."""
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'target'
        
    def prepare_data(self, df: pd.DataFrame, target_type: str = 'buy_signal') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Veriyi model eğitimi için hazırlar.
        
        Args:
            df (pd.DataFrame): Özelliklerle DataFrame
            target_type (str): Hedef değişken türü ('buy_signal', 'sell_signal', 'strong_buy')
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Özellikler ve hedef
        """
        # Hedef değişkeni belirle
        if target_type == 'buy_signal':
            target = df['Strong_Buy_Signal']
        elif target_type == 'sell_signal':
            target = df['Strong_Sell_Signal']
        elif target_type == 'rsi_buy':
            target = df['RSI_Buy_Signal']
        elif target_type == 'macd_buy':
            target = df['MACD_Buy_Signal']
        else:
            target = df['Strong_Buy_Signal']
        
        # Özellik sütunlarını belirle
        feature_columns = [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Position',
            'MA_10', 'MA_20', 'MA_50', 'MA_200',
            'Stoch_K', 'Stoch_D', 'ATR', 'OBV', 'VWAP', 'MFI',
            'Price_Change', 'Volume_Change', 'High_Low_Ratio', 'Close_Open_Ratio',
            'RSI_Overbought', 'RSI_Oversold', 'RSI_Buy_Signal',
            'MACD_Buy_Signal', 'MACD_Bullish_Divergence',
            'BB_Buy_Signal', 'BB_Sell_Signal',
            'MA_Cross_Buy', 'MA_Cross_Sell',
            'Stoch_Overbought', 'Stoch_Oversold',
            'Volume_Confirmation'
        ]
        
        # Mevcut sütunları kontrol et
        available_features = [col for col in feature_columns if col in df.columns]
        self.feature_columns = available_features
        
        # Özellikler ve hedef
        X = df[available_features].fillna(0)
        y = target.fillna(0)
        
        return X, y
    
    def create_models(self) -> Dict[str, Any]:
        """
        Makine öğrenmesi modellerini oluşturur.
        
        Returns:
            Dict[str, Any]: Model sözlüğü
        """
        models = {
            'Bagged_Tree': BaggingClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=10, random_state=42),
                n_estimators=100,
                random_state=42
            ),
            'Random_Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'Decision_Tree': DecisionTreeClassifier(
                max_depth=10,
                random_state=42
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5
            ),
            'Neural_Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            ),
            'Logistic_Regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
        
        return models
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Dict]:
        """
        Tüm modelleri eğitir ve değerlendirir.
        
        Args:
            X (pd.DataFrame): Özellikler
            y (pd.Series): Hedef değişken
            test_size (float): Test seti oranı
            
        Returns:
            Dict[str, Dict]: Model performansları
        """
        # Veriyi böl
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Veriyi ölçeklendir
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Modelleri oluştur
        models = self.create_models()
        results = {}
        
        for name, model in models.items():
            print(f"{name} modeli eğitiliyor...")
            
            try:
                # Modeli eğit
                if name == 'Neural_Network':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Performans metriklerini hesapla
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # Çapraz doğrulama
                cv_scores = cross_val_score(model, X_train, y_train, cv=10)
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred,
                    'y_test': y_test
                }
                
                print(f"{name} - Doğruluk: {accuracy:.4f}, F1: {f1:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                print(f"{name} modeli eğitilirken hata: {e}")
                continue
        
        self.models = results
        return results
    
    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, Any]:
        """
        En iyi performans gösteren modeli döndürür.
        
        Args:
            metric (str): Değerlendirme metriği
            
        Returns:
            Tuple[str, Any]: En iyi model adı ve modeli
        """
        if not self.models:
            raise ValueError("Henüz model eğitilmemiş")
        
        best_model_name = max(self.models.keys(), 
                            key=lambda x: self.models[x][metric])
        best_model = self.models[best_model_name]['model']
        
        return best_model_name, best_model
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, model_name: str = 'Bagged_Tree') -> Dict:
        """
        Hiperparametre optimizasyonu yapar.
        
        Args:
            X (pd.DataFrame): Özellikler
            y (pd.Series): Hedef değişken
            model_name (str): Optimize edilecek model adı
            
        Returns:
            Dict: En iyi parametreler
        """
        param_grids = {
            'Bagged_Tree': {
                'n_estimators': [50, 100, 200],
                'base_estimator__max_depth': [5, 10, 15]
            },
            'Random_Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            },
            'Decision_Tree': {
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
        
        if model_name not in param_grids:
            print(f"{model_name} için hiperparametre grid'i tanımlanmamış")
            return {}
        
        # Model oluştur
        models = self.create_models()
        model = models[model_name]
        
        # Grid search
        grid_search = GridSearchCV(
            model, 
            param_grids[model_name], 
            cv=5, 
            scoring='f1',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        print(f"En iyi parametreler: {grid_search.best_params_}")
        print(f"En iyi skor: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def save_model(self, model_name: str, filename: str):
        """
        Modeli dosyaya kaydeder.
        
        Args:
            model_name (str): Kaydedilecek model adı
            filename (str): Dosya adı
        """
        try:
            if model_name in self.models:
                model = self.models[model_name]['model']
                joblib.dump(model, f"models/{filename}.pkl")
                joblib.dump(self.scaler, f"models/{filename}_scaler.pkl")
                print(f"Model {filename} olarak kaydedildi")
            else:
                print(f"Model {model_name} bulunamadı")
                
        except Exception as e:
            print(f"Model kaydetme hatası: {e}")
    
    def load_model(self, filename: str) -> Tuple[Any, StandardScaler]:
        """
        Kaydedilmiş modeli yükler.
        
        Args:
            filename (str): Model dosya adı
            
        Returns:
            Tuple[Any, StandardScaler]: Model ve scaler
        """
        try:
            model = joblib.load(f"models/{filename}.pkl")
            scaler = joblib.load(f"models/{filename}_scaler.pkl")
            return model, scaler
            
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            return None, None
    
    def predict(self, model, X: pd.DataFrame, use_scaler: bool = True) -> np.ndarray:
        """
        Model ile tahmin yapar.
        
        Args:
            model: Eğitilmiş model
            X (pd.DataFrame): Tahmin edilecek veriler
            use_scaler (bool): Scaler kullanılıp kullanılmayacağı
            
        Returns:
            np.ndarray: Tahminler
        """
        try:
            if use_scaler:
                X_scaled = self.scaler.transform(X)
                return model.predict(X_scaled)
            else:
                return model.predict(X)
                
        except Exception as e:
            print(f"Tahmin hatası: {e}")
            return np.array([])
    
    def evaluate_model(self, model_name: str) -> Dict:
        """
        Model performansını detaylı değerlendirir.
        
        Args:
            model_name (str): Değerlendirilecek model adı
            
        Returns:
            Dict: Detaylı performans metrikleri
        """
        if model_name not in self.models:
            print(f"Model {model_name} bulunamadı")
            return {}
        
        model_data = self.models[model_name]
        y_test = model_data['y_test']
        y_pred = model_data['y_pred']
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        evaluation = {
            'accuracy': model_data['accuracy'],
            'precision': model_data['precision'],
            'recall': model_data['recall'],
            'f1_score': model_data['f1_score'],
            'cv_mean': model_data['cv_mean'],
            'cv_std': model_data['cv_std'],
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        return evaluation
    
    def print_model_comparison(self):
        """Tüm modellerin performansını karşılaştırır."""
        if not self.models:
            print("Henüz model eğitilmemiş")
            return
        
        print("\n" + "="*80)
        print("MODEL PERFORMANS KARŞILAŞTIRMASI")
        print("="*80)
        
        comparison_data = []
        for name, data in self.models.items():
            comparison_data.append({
                'Model': name,
                'Doğruluk': f"{data['accuracy']:.4f}",
                'Kesinlik': f"{data['precision']:.4f}",
                'Duyarlılık': f"{data['recall']:.4f}",
                'F1 Skoru': f"{data['f1_score']:.4f}",
                'CV Ortalama': f"{data['cv_mean']:.4f}",
                'CV Std': f"{data['cv_std']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        print("="*80)

# Test fonksiyonu
if __name__ == "__main__":
    # Test verisi oluştur
    np.random.seed(42)
    n_samples = 1000
    
    # Simüle edilmiş özellikler
    X = pd.DataFrame({
        'RSI': np.random.uniform(0, 100, n_samples),
        'MACD': np.random.uniform(-2, 2, n_samples),
        'BB_Position': np.random.uniform(0, 1, n_samples),
        'Volume_Confirmation': np.random.choice([0, 1], n_samples),
        'Price_Change': np.random.uniform(-0.1, 0.1, n_samples)
    })
    
    # Simüle edilmiş hedef (alım sinyali)
    y = ((X['RSI'] < 30) & (X['MACD'] > 0) & (X['BB_Position'] < 0.2)).astype(int)
    
    # Model eğitimi
    ml_models = CryptoMLModels()
    results = ml_models.train_models(X, y)
    
    # Model karşılaştırması
    ml_models.print_model_comparison()
    
    # En iyi modeli bul
    best_model_name, best_model = ml_models.get_best_model()
    print(f"\nEn iyi model: {best_model_name}")
    
    # Modeli kaydet
    ml_models.save_model(best_model_name, "best_crypto_model") 