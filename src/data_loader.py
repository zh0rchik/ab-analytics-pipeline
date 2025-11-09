import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
    
    def load_raw(self) -> pd.DataFrame:
        """Загрузка сырых данных из CSV файла"""
        try:
            df = pd.read_csv(self.file_path)
            logger.info(f"Данные успешно загружены. Размер: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Предобработка и очистка данных"""
        # Сохраняем копию для отслеживания изменений
        df_processed = df.copy()
        
        # Приведение типов
        df_processed['user id'] = df_processed['user id'].astype(int)
        
        # Стандартизация названий колонок
        df_processed.columns = [c.strip().lower().replace(' ', '_') for c in df_processed.columns]
        
        logger.info("Данные предобработаны")
        logger.info(f"Новые названия колонок: {list(df_processed.columns)}")
        
        return df_processed
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Валидация данных"""
        try:
            # Проверка уникальности user_id
            assert df['user_id'].is_unique, 'user_id not unique!'
            
            # Проверка на пропущенные значения
            missing_values = df.isnull().sum()
            if missing_values.any():
                logger.warning(f"Найдены пропущенные значения: {missing_values[missing_values > 0].to_dict()}")
            
            # Проверка существования обязательных колонок
            required_columns = ['user_id', 'test_group', 'converted']
            for col in required_columns:
                assert col in df.columns, f'Отсутствует обязательная колонка: {col}'
            
            logger.info("Валидация данных пройдена")
            return True
            
        except AssertionError as e:
            logger.error(f"Ошибка валидации: {e}")
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка при валидации: {e}")
            return False
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Полный пайплайн загрузки и подготовки данных"""
        # Загрузка
        raw_df = self.load_raw()
        
        # Предобработка
        processed_df = self.preprocess_data(raw_df)
        
        # Валидация
        if self.validate_data(processed_df):
            self.df = processed_df
            return processed_df
        else:
            raise ValueError("Данные не прошли валидацию")
    
    def get_basic_info(self) -> Dict:
        """Базовая информация о данных"""
        if self.df is None:
            return {}
        
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'data_types': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'groups_distribution': self.df['test_group'].value_counts().to_dict(),
            'conversion_rate': self.df['converted'].value_counts().to_dict(),
            'memory_usage_mb': round(self.df.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        return info

# Функция для быстрой загрузки
def load_raw(path: str) -> pd.DataFrame:
    """Быстрая загрузка сырых данных"""
    return pd.read_csv(path)