import pyarrow as pa
import numpy as np

def numpy_dtype_to_arrow_dtype(np_dtype: np.dtype) -> pa.DataType:
  """
  Maps a NumPy dtype to an Apache Arrow data type.

  Parameters:
  - np_dtype: A NumPy data type (np.dtype).

  Returns:
  - An Apache Arrow data type (pa.DataType).
  """
  if np_dtype == np.dtype('int32'):
    return pa.int32()
  elif np_dtype == np.dtype('int64'):
    return pa.int64()
  elif np_dtype == np.dtype('float32'):
    return pa.float32()
  elif np_dtype == np.dtype('float64'):
    return pa.float64()
  elif np_dtype == np.dtype('bool'):
    return pa.bool_()
  elif np_dtype == np.dtype('object') or np_dtype.type == np.str_ or np_dtype.kind == 'U':
    return pa.string()
  elif np_dtype == np.dtype('datetime64[ms]'):
    return pa.timestamp('ms')
  else:
    raise ValueError(f"Unsupported NumPy dtype: {np_dtype}")
