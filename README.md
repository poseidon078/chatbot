# 💬 Vibe to Fashion Attribute Mapper

This Streamlit app helps shoppers find clothing items that match their **fashion vibe**.  
Just describe the style you're going for (e.g., "flowy summer dress for a beach vacation"), and the app will:

1. **Extract fashion attributes** like fabric, fit, color, occasion, etc.
2. **Match those attributes** against a real product catalog (`Apparels_shared.xlsx`).
3. **Display the top matching items** based on key-wise semantic similarity.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
