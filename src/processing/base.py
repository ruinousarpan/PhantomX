from typing import Any, Dict, List, Optional, Type, Union, Callable
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class ProcessingResult:
    """Represents the result of a data processing operation."""
    
    def __init__(
        self,
        success: bool,
        data: Optional[Any] = None,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.data = data
        self.errors = errors or []
        self.warnings = warnings or []
        self.metadata = metadata or {}

    def __bool__(self) -> bool:
        return self.success

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata
        }

class DataProcessor(ABC):
    """Abstract base class for all data processors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the processor configuration."""
        pass

    @abstractmethod
    def process(self, data: Any) -> ProcessingResult:
        """Process the given data."""
        pass

    def process_batch(self, data_list: List[Any]) -> List[ProcessingResult]:
        """Process a batch of data items."""
        return [self.process(item) for item in data_list]

class DataFrameProcessor(DataProcessor):
    """Base class for pandas DataFrame processors."""
    
    def _validate_config(self) -> None:
        """Validate the processor configuration."""
        pass

    def process(self, data: pd.DataFrame) -> ProcessingResult:
        """Process a pandas DataFrame."""
        if not isinstance(data, pd.DataFrame):
            return ProcessingResult(
                success=False,
                errors=[f"Expected pandas DataFrame, got {type(data)}"]
            )
        
        try:
            processed_data = self._process_dataframe(data)
            return ProcessingResult(success=True, data=processed_data)
        except Exception as e:
            return ProcessingResult(
                success=False,
                errors=[f"Error processing DataFrame: {str(e)}"]
            )

    @abstractmethod
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a pandas DataFrame. To be implemented by subclasses."""
        pass

class DataPipeline:
    """A pipeline of data processors that can be applied sequentially."""
    
    def __init__(self, processors: Optional[List[DataProcessor]] = None):
        self.processors = processors or []

    def add_processor(self, processor: DataProcessor) -> 'DataPipeline':
        """Add a processor to the pipeline."""
        self.processors.append(processor)
        return self

    def process(self, data: Any) -> ProcessingResult:
        """Process data through the entire pipeline."""
        if not self.processors:
            return ProcessingResult(
                success=True,
                data=data,
                warnings=["No processors in pipeline"]
            )

        current_data = data
        all_errors = []
        all_warnings = []
        metadata = {}

        for i, processor in enumerate(self.processors):
            try:
                result = processor.process(current_data)
                
                if not result.success:
                    all_errors.extend([f"Processor {i} ({processor.__class__.__name__}): {err}" for err in result.errors])
                    return ProcessingResult(
                        success=False,
                        errors=all_errors,
                        warnings=all_warnings,
                        metadata=metadata
                    )
                
                current_data = result.data
                all_warnings.extend([f"Processor {i} ({processor.__class__.__name__}): {warn}" for warn in result.warnings])
                metadata.update(result.metadata)
            except Exception as e:
                all_errors.append(f"Processor {i} ({processor.__class__.__name__}) failed with error: {str(e)}")
                return ProcessingResult(
                    success=False,
                    errors=all_errors,
                    warnings=all_warnings,
                    metadata=metadata
                )

        return ProcessingResult(
            success=True,
            data=current_data,
            warnings=all_warnings,
            metadata=metadata
        ) 