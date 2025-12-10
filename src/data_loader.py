"""
Модуль для загрузки, предобработки и валидации данных для A/B тестирования.
"""
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Класс для загрузки и подготовки данных из CSV файла.
    
    Attributes:
        file_path (str): Путь к CSV файлу.
        df (pd.DataFrame | None): Обработанный и валидированный DataFrame.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df: pd.DataFrame | None = None

    def load_raw(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.file_path)
            logger.info(f"Данные успешно загружены из {self.file_path}. Размер: {df.shape}")
            return df
        except FileNotFoundError as e:
            logger.error(f"Ошибка: файл не найден по пути: {self.file_path}")
            raise e
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        df_processed.columns = [c.strip().lower().replace(' ', '_') for c in df_processed.columns]
        logger.info(f"Названия колонок стандартизированы: {list(df_processed.columns)}")
        if 'user_id' in df_processed.columns:
            df_processed['user_id'] = df_processed['user_id'].astype(int)
            logger.info("Тип колонки 'user_id' изменен на int.")
        return df_processed

    def validate_data(self, df: pd.DataFrame) -> bool:
        try:
            required_columns = ['user_id', 'test_group', 'converted']
            for col in required_columns:
                assert col in df.columns, f'Отсутствует обязательная колонка: {col}'
            assert df['user_id'].is_unique, 'user_id не уникален!'
            missing_values = df.isnull().sum()
            if missing_values.any():
                logger.warning(f"Найдены пропущенные значения: {missing_values[missing_values > 0].to_dict()}")
            logger.info("Валидация данных успешно пройдена.")
            return True
        except AssertionError as e:
            logger.error(f"Ошибка валидации данных: {e}")
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка при валидации: {e}")
            return False

    def load_and_prepare_data(self) -> pd.DataFrame:
        logger.info("Запуск полного пайплайна подготовки данных...")
        raw_df = self.load_raw()
        processed_df = self.preprocess_data(raw_df)

        if self.validate_data(processed_df):
            self.df = processed_df
            logger.info("Пайплайн завершен успешно.")
            return self.df
        else:
            raise ValueError("Данные не прошли валидацию, пайплайн остановлен.")

    def get_info(self) -> Dict: # <--- ИЗМЕНЕНИЕ ЗДЕСЬ
        """
        Возвращает базовую информацию о загруженных данных.
        """
        if self.df is None:
            logger.warning("DataFrame не загружен. Вызовите load_and_prepare_data() сначала.")
            return {}

        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'data_types': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'groups_distribution': self.df['test_group'].value_counts().to_dict(),
            'conversion_rate_overall': self.df['converted'].value_counts(normalize=True).to_dict(),
            'memory_usage_mb': round(self.df.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        return info

def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)