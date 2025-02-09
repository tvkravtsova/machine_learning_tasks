import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Tuple, List, Dict

def load_data(file_path: str) -> pd.DataFrame:
    """
    Завантажує дані з CSV-файлу.
    
    Args:
        file_path (str): Шлях до CSV-файлу.
    
    Returns:
        pd.DataFrame: Завантажений датафрейм.
    """
    return pd.read_csv(file_path)

def split_data(df: pd.DataFrame, target_col: str, test_size = 0.25, random_state = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Розділяє дані на тренувальні та валідаційні набори.
    
    Args:
        df (pd.DataFrame): Вхідний датафрейм.
        target_col (str): Назва цільового стовпця.
        test_size (float): Розмір тестової вибірки.
        random_state (int): Значення для відтворюваності.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
        Тренувальні та валідаційні дані.
    """
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])
    input_cols = list(df.columns)[3:-1] #Вибір ознак
    return train_df[input_cols], val_df[input_cols], train_df[target_col], val_df[target_col], input_cols

def scale_numeric_features(train_inputs: pd.DataFrame, val_inputs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Масштабує числові ознаки за допомогою MinMaxScaler.
    
    Args:
        train_inputs (pd.DataFrame): Тренувальні дані.
        val_inputs (pd.DataFrame): Валідаційні дані.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]: 
        Масштабовані тренувальні та валідаційні дані, а також використаний скейлер.
    """
    numeric_cols = train_inputs.select_dtypes(include=[np.number]).columns.tolist()
    scaler = MinMaxScaler().fit(train_inputs[numeric_cols])
    train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
    return train_inputs, val_inputs, scaler

def scale_numeric_features(train_inputs: pd.DataFrame, val_inputs: pd.DataFrame, scale: bool) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Масштабує числові ознаки за допомогою MinMaxScaler, якщо параметр scale має значення True.
    
    Args:
        train_inputs (pd.DataFrame): Тренувальні ознаки.
        val_inputs (pd.DataFrame): Валідаційні ознаки.
        scale (bool): Чи застосовувати масштабування.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
        Масштабовані тренувальні та валідаційні ознаки, а також використаний скейлер.
    """
    numeric_cols = train_inputs.select_dtypes(include=[np.number]).columns.tolist()
    scaler = None
    if scale:
        scaler = MinMaxScaler().fit(train_inputs[numeric_cols])
        train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
        val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
    return train_inputs, val_inputs, scaler


def encode_categorical_features(train_inputs: pd.DataFrame, val_inputs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """
    Кодує категоріальні ознаки за допомогою OneHotEncoder.
    
    Args:
        train_inputs (pd.DataFrame): Тренувальні дані.
        val_inputs (pd.DataFrame): Валідаційні дані.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]: 
        Закодовані тренувальні та валідаційні дані, а також енкодер.
    """
    categorical_cols = train_inputs.select_dtypes(include=['object']).columns.tolist()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    
    train_encoded = pd.DataFrame(encoder.transform(train_inputs[categorical_cols]), columns=encoded_cols, index=train_inputs.index)
    val_encoded = pd.DataFrame(encoder.transform(val_inputs[categorical_cols]), columns=encoded_cols, index=val_inputs.index)
    
    train_inputs = pd.concat([train_inputs.drop(columns=categorical_cols), train_encoded], axis=1)
    val_inputs = pd.concat([val_inputs.drop(columns=categorical_cols), val_encoded], axis=1)
    
    return train_inputs, val_inputs, encoder
    
def preprocess_data(raw_df: pd.DataFrame, scaler_numeric: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Повна обробка даних: розділення, масштабування (опційно) та кодування.
    
    Args:
        raw_df (pd.DataFrame): Сирі дані.
        scaler_numeric (bool): Чи застосовувати масштабування числових ознак.
    
    Returns:
        Dict[str, pd.DataFrame]: Оброблені тренувальні та валідаційні дані, скейлер і енкодер.
    """
    X_train, X_val, train_targets, val_targets, input_cols = split_data(raw_df, target_col='Exited')
    X_train, X_val, scaler = scale_numeric_features(X_train, X_val, scaler_numeric)
    X_train, X_val, encoder = encode_categorical_features(X_train, X_val)
    
    return {
        'X_train': X_train, 'X_val': X_val,
        'train_targets': train_targets, 'val_targets': val_targets,
        'scaler': scaler, 'encoder': encoder,
        'input_cols': input_cols
    }

def preprocess_new_data(new_data: pd.DataFrame, input_cols: list, scaler: MinMaxScaler, encoder: OneHotEncoder) -> pd.DataFrame:
    """
    Обробляє нові дані за допомогою раніше навченого скейлера та енкодера.
    
    Args:
        new_data (pd.DataFrame): Нові дані для обробки.
        input_cols (list): Список колонок, які потрібно обробити.
        scaler (MinMaxScaler): Раніше навчений скейлер.
        encoder (OneHotEncoder): Раніше навчений енкодер.
    
    Returns:
        pd.DataFrame: Оброблені нові дані.
    """
    # Розділяємо input_cols на числові та категоріальні колонки
    numeric_cols = [col for col in input_cols if new_data[col].dtype in [np.int64, np.float64]]
    categorical_cols = [col for col in input_cols if new_data[col].dtype == 'object']
    
    # Масштабуємо числові колонки (якщо є скейлер)
    if scaler and numeric_cols:
        new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])
    
    # Кодуємо категоріальні колонки за допомогою навченого енкодера
    if categorical_cols:
        encoded_new = encoder.transform(new_data[categorical_cols])
        encoded_df = pd.DataFrame(encoded_new, columns=encoder.get_feature_names_out(categorical_cols), index=new_data.index)
        # Об'єднуємо числові та категоріальні дані
        return pd.concat([new_data[numeric_cols], encoded_df], axis=1)
    
    return new_data

