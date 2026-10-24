# Collect product details, call the Flask API, and display or download sales predictions.
import os
import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.environ.get('BACKEND_URL', 'http://127.0.0.1:7860').rstrip('/')
st.set_page_config(page_title='SuperKart | Sales', page_icon=':shopping_cart:', layout='wide')
st.title('SuperKart')
st.subheader('Sales Predictions')

def api_request(method, endpoint, **kwargs):
    response = requests.request(method, f'{BACKEND_URL}{endpoint}', timeout=30, **kwargs)
    try:
        body = response.json()
    except ValueError:
        raise ValueError('The API returned an unreadable response.')
    if not response.ok:
        raise ValueError(body.get('error', f'API error: {response.status_code}'))
    return body

try:
    schema = api_request('GET', '/v1/schema')
except (requests.RequestException, ValueError) as error:
    st.error(f'Backend unavailable: {error}')
    st.stop()

single_tab, batch_tab = st.tabs(['Single Product', 'Batch Predictions'])
with single_tab:
    with st.form('single_product'):
        columns = st.columns(2)
        payload = {}
        for index, feature in enumerate(schema['features']):
            label = feature.replace('_', ' ')
            with columns[index % 2]:
                if feature in schema['categorical']:
                    options = schema['categorical'][feature]
                    payload[feature] = st.selectbox(label, options, index=options.index(schema['defaults'][feature]))
                elif feature == 'Store_Age_Years':
                    payload[feature] = st.number_input('Store Age (2026 Reference)', min_value=0, value=int(schema['defaults'][feature]), step=1)
                elif feature == 'Product_Allocated_Area':
                    payload[feature] = st.number_input(label, min_value=0.0, max_value=1.0, value=float(schema['defaults'][feature]), step=0.001, format='%.3f')
                else:
                    payload[feature] = st.number_input(label, min_value=0.0, value=float(schema['defaults'][feature]))
        submitted = st.form_submit_button('Predict Sales', type='primary')
    if submitted:
        try:
            result = api_request('POST', '/v1/predict', json=payload)
            st.metric('Predicted Product Store Sales Total', f"{result['Sales']:,.2f}")
        except (requests.RequestException, ValueError) as error:
            st.error(str(error))

with batch_tab:
    example = pd.DataFrame([schema['defaults']])
    st.download_button('Download CSV Template', example.to_csv(index=False), 'superkart_template.csv', 'text/csv')
    uploaded_file = st.file_uploader('Product and Store Data', type=['csv'])
    if uploaded_file is not None and st.button('Predict Batch', type='primary'):
        try:
            raw = uploaded_file.getvalue()
            from io import BytesIO
            input_frame = pd.read_csv(BytesIO(raw))
            result = api_request('POST', '/v1/predictbatch', files={'file': (uploaded_file.name, raw, 'text/csv')})
            input_frame['Predicted_Sales'] = [result[str(index)] for index in range(len(input_frame))]
            st.session_state['batch_results'] = input_frame
            st.session_state['batch_source'] = raw
        except (requests.RequestException, ValueError, KeyError) as error:
            st.session_state.pop('batch_results', None)
            st.error(str(error))
    if uploaded_file is not None and st.session_state.get('batch_source') == uploaded_file.getvalue() and 'batch_results' in st.session_state:
        results = st.session_state['batch_results']
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.download_button('Download Predictions', results.to_csv(index=False), 'superkart_predictions.csv', 'text/csv')
