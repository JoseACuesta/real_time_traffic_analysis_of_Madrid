import streamlit as st
from pathlib import Path

st.set_page_config(page_title='Real-time analysis of traffic data in Chamberí', page_icon='🚗🛵📈')

rfr_page = st.Page(page=Path('pages/rfr_page.py'), title='Random Forest Regressor model', icon='🌲')
xgb_page = st.Page(page=Path('pages/xgb_page.py'), title='XGBoost Regressor model', icon='🚀')


pg = st.navigation(pages=[rfr_page, xgb_page])

pg.run()