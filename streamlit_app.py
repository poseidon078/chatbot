import os, streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer, util
import json
import sys
import pandas as pd
import re

# Load catalog
df = pd.read_excel("Apparels_shared.xlsx")

# Ensure all column names are lowercase for matching
df.columns = [c.lower() for c in df.columns]

# Lowercase all cell values for string comparison
df = df.map(lambda x: str(x).lower() if pd.notna(x) else x)

# Match function
def is_match(row, final):
    for key, val in final.items():
        if key not in row or pd.isna(row[key]):
            continue
        cell = row[key]
        if isinstance(val, list):
            if any(v.lower() in cell for v in val):
                return True
        elif isinstance(val, str):
            if val.lower() in cell:
                return True
    return False

def match_score(row, final):
    score = 0
    for key, val in final.items():
        if key not in row or pd.isna(row[key]):
            continue
        cell = row[key]
        if isinstance(val, list):
            score += sum(bool(re.search(rf"\b{re.escape(v.lower())}\b", cell)) for v in val)
        elif isinstance(val, str):
            score += bool(re.search(rf"\b{re.escape(val.lower())}\b", cell))
    return score


# Patch to avoid torch.classes bug in Streamlit Cloud
sys.modules['torch.classes'] = None

# Load API key
GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
client = Groq()

model = SentenceTransformer('all-MiniLM-L6-v2', device = 'cpu')

# Vibe-based examples
typical_examples = {
    "elevated date-night shine for tops": {"fit": "Body hugging", "fabric": ["Satin", "Velvet", "Silk"]},
    "comfy lounge tees": {"fit": "Relaxed", "sleeve_length": "Short sleeves"},
    "office-ready polish shirts": {"fabric": "Cotton poplin", "neckline": "Collar"},
    "flowy dresses for garden-party": {"fit": "Relaxed", "fabric": ["Chiffon", "Linen"], "sleeve_length": "Short flutter sleeves", "color_or_print": "Pastel floral", "occasion": "Party"},
    "elevated evening glam dresses": {"fit": "Body hugging", "fabric": ["Satin", "Silk"], "sleeve_length": "Sleeveless", "color_or_print": "Sapphire blue", "occasion": "Party", "length": ["Midi", "Mini", "Short"]},
    "beachy vacay dress": {"fit": "Relaxed", "fabric": "Linen", "sleeve_length": "Spaghetti straps", "color_or_print": "Seafoam green", "occasion": "Vacation"},
    "dresses for retro 70s look": {"fit": "Body hugging", "fabric": "Stretch crepe", "sleeve_length": "Cap sleeves", "color_or_print": "Geometric print"},
    "pastel": {"color_or_print": ["pastel pink", "pastel yellow"]},
    "floral": {"color_or_print": ["floral print"]},
    "bold": {"color_or_print": ["ruby red", "cobalt blue"]},
    "neutral": {"color_or_print": ["sand beige", "off-white", "white"]},
    "flowy": {"fit": "relaxed"},
    "bodycon": {"fit": "body hugging"},
    "breathable": {"fabric": "linen"},
    "summer": {"fabric": "linen"},
    "luxurious": {"fabric": "velvet"},
    "party": {"fabric": "velvet"},
    "metallic": {"fabric": "lamé"},
    "sleek": {"fit": "slim"},
    "retro '70's flare vibe": {"fit": "sleek and straight", "type": "flared"},
    "breathable summer": {"fabric": "linen"}
}

example_embeddings = model.encode(list(typical_examples.keys()), convert_to_tensor=True)


def extract_json_from_response(response):
    try:
        start = response.index('{')
        end = response.rindex('}') + 1
        json_str = response[start:end]
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"Could not extract valid JSON from response. Reason: {e}")


def parse_stream(stream):
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def missing_attributes(attr_dict):
    expected = ["category", "fabric", "color_or_print", "sleeve_length", "occasion", "neckline", "length", "fit", "size", "price"]
    return [attr for attr in expected if attr not in attr_dict]


def return_attribute_json(user_query):
    user_embedding = model.encode(user_query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(user_embedding, example_embeddings)[0]
    top_indices = cosine_scores.argsort(descending=True)[:3]
    best_index = top_indices[0]
    best_match = list(typical_examples.keys())[best_index]
    confidence = cosine_scores[best_index].item()

    if confidence > 0.6:
        st.session_state.chat_history.append({"role": "user", "content": user_query, "user_query": user_query})
        # with st.chat_message("assistant"):
            # st.json(typical_examples[best_match])
        st.session_state.chat_history.append({"role": "assistant", "content": json.dumps(typical_examples[best_match]), "user_query": json.dumps(typical_examples[best_match]), "attributes": typical_examples[best_match]})
        st.session_state.partial_json = typical_examples[best_match]
    else:
        top_k = {k: typical_examples[k] for i, k in enumerate(typical_examples.keys()) if i in top_indices}
        prior = "\n".join([f"- {k}: {json.dumps(v)}" for k, v in top_k.items()])
        prompt = f"""
Map the user query delimited by triple backticks ```{user_query}``` to structured fashion attributes.
Don't map if unsure.
Here are similar examples:
{prior}
Return only JSON with keys ONLY if not null:
- category, fabric, color_or_print, sleeve_length, occasion,
  neckline, length, fit, size, price
Only return the JSON and nothing else.
        """
        st.session_state.chat_history.append({"role": "user", "content": prompt, "user_query": user_query})

        with st.chat_message("assistant"):
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history]
            stream = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=history,
                stream=True
            )
            response = st.write_stream(parse_stream(stream))

        try:
            parsed = extract_json_from_response(response)
        except Exception as e:
            st.error(f"❌ JSON Parse Error: {e}")
            st.markdown(f"```\n{response}\n```")
            return

        st.session_state.partial_json = parsed
    missing = missing_attributes(st.session_state.partial_json)
    

    if missing:
        followup = f"Can you help me fill in these missing details: {', '.join(missing)}?"
        with st.chat_message("assistant"):
            st.markdown(followup)
        st.session_state.chat_history.append({"role": "assistant", "content": followup, "user_query": followup})


# UI starts
st.title("👗 Vibe to Fashion Attribute Mapper")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "Hi! Tell me the vibe you're going for, and I’ll help find matching attributes."}]

for msg in st.session_state.chat_history:
    content = msg.get("user_query", msg["content"])
    # Skip displaying JSON messages
    if content.strip().startswith("{") and content.strip().endswith("}"):
        continue
    st.chat_message(msg["role"]).markdown(content)

if user_input := st.chat_input("Describe your vibe:"):
    # Detect follow-up response
    if "partial_json" in st.session_state and st.session_state.chat_history[-1]["role"] == "assistant" and "missing details" in st.session_state.chat_history[-1]["content"].lower():
        st.session_state.chat_history.append({"role": "user", "content": user_input, "user_query": user_input})

        prompt = f"""
We previously collected these attributes: {json.dumps(st.session_state.partial_json, indent=2)}.
The user added: ```{user_input}```.
Please return a single combined JSON with these keys if available:
- category, fabric, color_or_print, sleeve_length, occasion,
  neckline, length, fit, size, price
Return only the final JSON.
        """

        st.session_state.chat_history.append({"role": "user", "content": prompt, "user_query": user_input})

        with st.chat_message("assistant"):
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history]
            stream = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=history,
                stream=True
            )
            response = st.write_stream(parse_stream(stream))

        try:
            parsed = extract_json_from_response(response)
        except Exception as e:
            st.error(f"❌ Follow-up JSON Error: {e}")
            st.markdown(f"```\n{response}\n```")
            # return

        st.session_state.partial_json.update(parsed)
        final = st.session_state.partial_json
        # st.json(final)

        st.session_state.chat_history.append({"role": "assistant", "content": json.dumps(final, indent=2), "user_query": json.dumps(final)})

        df["match_score"] = df.apply(lambda row: match_score(row, final), axis=1)
        matched_df = df[df["match_score"] > 0].sort_values(by="match_score", ascending=False)
        N = 2
        
        # Show result
        if not matched_df.empty:
            st.success(f"Found matching items:")
            st.dataframe(matched_df.head(N))
        else:
            st.warning("No matching items found.")
    else:
        return_attribute_json(user_input)
