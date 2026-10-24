# Load the saved model and validate inputs before returning single or batch sales predictions.
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest, UnsupportedMediaType

ROOT = Path(__file__).resolve().parent
model = joblib.load(ROOT / 'superkart_model.joblib')
schema = json.loads((ROOT / 'schema.json').read_text(encoding='utf-8'))
superkart_api = Flask('SuperKart')
superkart_api.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
ALIASES = {'Product_Id_char': 'Product_Category', 'Product_Type_Category': 'Product_Category_Type'}

# failuremode-handling start
# Reject an invalid file structure before checking its individual products.
def prepare_columns(frame):
    if frame.empty:
        raise ValueError('Provide at least one product.')
    for old, new in ALIASES.items():
        if old in frame.columns and new in frame.columns:
            raise ValueError(f'Provide either {old} or {new}, not both.')
    frame = frame.rename(columns=ALIASES)
    missing = sorted(set(schema['features']) - set(frame.columns))
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(missing))
    return frame.loc[:, schema['features']].copy()

def prepare_input(frame):
    frame = prepare_columns(frame)
    # ailuremode-handling end
    frame['Product_Sugar_Content'] = frame['Product_Sugar_Content'].replace({'reg': 'Regular'})
    if frame.isna().any().any():
        raise ValueError('Input contains missing values.')
    for column in schema['numeric']:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
        if not np.isfinite(frame[column]).all() or (frame[column] < 0).any():
            raise ValueError(f'{column} must contain finite, non-negative numbers.')
    if (frame['Product_Allocated_Area'] > 1).any():
        raise ValueError('Product_Allocated_Area must be between 0 and 1.')
    if (frame['Store_Age_Years'] % 1 != 0).any():
        raise ValueError('Store_Age_Years must contain whole years, relative to 2026.')
    for column, labels in schema['categorical'].items():
        invalid = frame.loc[~frame[column].isin(labels), column].astype(str).unique().tolist()
        if invalid:
            raise ValueError(f'Unsupported {column}: {invalid}. Allowed: {labels}')
    return frame

@superkart_api.get('/')
def home():
    return jsonify({'status': 'ok', 'model': schema['model_name']})

@superkart_api.get('/v1/schema')
def get_schema():
    return jsonify(schema)

@superkart_api.post('/v1/predict')
def predict_sales():
    payload = request.get_json()
    if not isinstance(payload, dict):
        raise ValueError('Send one product as a JSON object.')
    frame = prepare_input(pd.DataFrame([payload]))
    return jsonify({'Sales': float(model.predict(frame)[0])})

@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    # failuremode-handling start
    # Keep one result per input row, even when some products cannot be predicted.
    if 'file' not in request.files:
        raise ValueError('Upload a CSV using the file field.')
    frame = prepare_columns(pd.read_csv(request.files['file']))
    results, valid_frames, valid_positions = [], [], []
    for position in range(len(frame)):
        result = {'row': position, 'status': 'rejected', 'predicted_sales': None, 'error': None}
        try:
            prepared = prepare_input(frame.iloc[[position]])
        except (ValueError, TypeError, OverflowError) as error:
            result['error'] = str(error)
        else:
            result['status'] = 'success'
            valid_frames.append(prepared)
            valid_positions.append(position)
        results.append(result)
    if valid_frames:
        predictions = model.predict(pd.concat(valid_frames, ignore_index=True))
        for position, prediction in zip(valid_positions, predictions):
            results[position]['predicted_sales'] = float(prediction)
    return jsonify(results)
    # ailuremode-handling end

@superkart_api.errorhandler(ValueError)
@superkart_api.errorhandler(BadRequest)
@superkart_api.errorhandler(UnsupportedMediaType)
def invalid_input(error):
    return jsonify({'error': str(error)}), 400

@superkart_api.errorhandler(413)
def oversized_file(error):
    return jsonify({'error': 'The CSV must be smaller than 5 MB.'}), 413

if __name__ == '__main__':
    superkart_api.run(host='127.0.0.1', port=7860, debug=False)
