import re
import logging
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

class InputValidator:
    def __init__(self):
        self._setup_logging()
        self._setup_patterns()
        
    def _setup_logging(self):
        """Setup secure logging"""
        logging.basicConfig(
            filename='data/validation.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('InputValidator')

    def _setup_patterns(self):
        """Setup validation patterns"""
        self.patterns = {
            'time': r'^\d{10}$',  # Unix timestamp
            'username': r'^[a-zA-Z0-9_@.-]{1,64}$',
            'computer': r'^[a-zA-Z0-9_@.-]{1,64}$',
            'auth_type': r'^[A-Z0-9_]{1,32}$',
            'logon_type': r'^[A-Z0-9_]{1,32}$',
            'auth_orientation': r'^[A-Z0-9_]{1,32}$',
            'success': r'^(true|false)$'
        }

    def sanitize_string(self, value: str) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            return str(value)
        # Remove any non-printable characters
        value = ''.join(char for char in value if char.isprintable())
        # Remove any potential SQL injection patterns
        value = re.sub(r'[\'";]', '', value)
        return value.strip()

    def validate_field(self, field: str, value: Any) -> bool:
        """Validate a single field against its pattern"""
        if field not in self.patterns:
            self.logger.warning(f"Unknown field type: {field}")
            return False

        if not isinstance(value, str):
            value = str(value)

        value = self.sanitize_string(value)
        pattern = self.patterns[field]
        
        if not re.match(pattern, value):
            self.logger.warning(f"Validation failed for {field}: {value}")
            return False
            
        return True

    def validate_row(self, row: Dict[str, Any]) -> bool:
        """Validate a single row of data"""
        for field, value in row.items():
            if not self.validate_field(field, value):
                return False
        return True

    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and sanitize an entire DataFrame"""
        # Create a copy to avoid modifying the original
        validated_df = df.copy()
        
        # Validate each column
        for column in validated_df.columns:
            if column in self.patterns:
                # Apply validation and sanitization
                validated_df[column] = validated_df[column].apply(
                    lambda x: self.sanitize_string(x) if self.validate_field(column, x) else np.nan
                )
        
        # Remove rows with any NaN values (failed validation)
        original_len = len(validated_df)
        validated_df = validated_df.dropna()
        removed_rows = original_len - len(validated_df)
        
        if removed_rows > 0:
            self.logger.warning(f"Removed {removed_rows} rows due to validation failures")
        
        return validated_df

    def validate_file_path(self, file_path: str) -> bool:
        """Validate file path for security"""
        # Check for path traversal attempts
        if '..' in file_path or file_path.startswith('/'):
            self.logger.warning(f"Invalid file path detected: {file_path}")
            return False
            
        # Check for allowed file extensions
        allowed_extensions = {'.csv', '.gz', '.json', '.pth'}
        if not any(file_path.endswith(ext) for ext in allowed_extensions):
            self.logger.warning(f"Invalid file extension in path: {file_path}")
            return False
            
        return True

    def validate_model_input(self, input_data: np.ndarray) -> bool:
        """Validate model input data"""
        if not isinstance(input_data, np.ndarray):
            self.logger.warning("Model input must be numpy array")
            return False
            
        # Check for NaN or infinite values
        if np.isnan(input_data).any() or np.isinf(input_data).any():
            self.logger.warning("Model input contains NaN or infinite values")
            return False
            
        # Check for reasonable value ranges
        if np.abs(input_data).max() > 1e6:
            self.logger.warning("Model input contains extreme values")
            return False
            
        return True 