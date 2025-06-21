import os, streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer, util
import json
import sys
import pandas as pd
import re

# Patch to avoid torch.classes bug in Streamlit Cloud
sys.modules['torch.classes'] = None

# Load API key
GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
client = Groq()


model = SentenceTransformer('all-MiniLM-L6-v2', device = 'cpu')

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

# Compute and store row-wise key → embedding dictionary
df["embedding_dict"] = None



def keywise_similarity(final_json, df, model):
    # Precompute final key embeddings

    for idx, row in df.iterrows():
        row_embeddings = {}
        for key in row.index:
            val = row[key]
            if pd.isna(val) or not isinstance(val, str) or val.strip() == "":
                continue
            row_embeddings[key] = model.encode(val, convert_to_tensor=True)
        df.at[idx, "embedding_dict"] = row_embeddings

    final_embeds = {}
    for key, val in final_json.items():
        if not val or val in ["", None, [], "null"]:
            continue
        if isinstance(val, list):
            text = ", ".join(val)
        else:
            text = str(val)
        final_embeds[key] = model.encode(text, convert_to_tensor=True)

    # Loop through rows and compute average cosine score
    scores = []
    for _, row in df.iterrows():
        row_embeds = row["embedding_dict"]
        if not isinstance(row_embeds, dict):
            scores.append(0)
            continue
        total_score = 0
        count = 0

        for key, final_embed in final_embeds.items():
            row_embed = row_embeds.get(key)
            if row_embed is not None:
                sim = util.cos_sim(final_embed, row_embed).item()
                total_score += sim
                count += 1

        avg_score = total_score / count if count > 0 else 0
        scores.append(avg_score)

    return scores

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


def extract_json_from_response(response: str):
    try:
        # Extract the first JSON-looking block
        match = re.search(r"{.*?}", response, re.DOTALL)
        if not match:
            return None

        json_str = match.group(0)

        # Fix unquoted strings in lists: ["value1", value2] → ["value1", "value2"]
        json_str = re.sub(r'(?<=\[)([^"\]]+?)(?=\])', lambda m: '"' + m.group(1).strip() + '"', json_str)

        # Escape any rogue backslashes (e.g., \ L)
        json_str = re.sub(r'\\(?![nrt"\\/bfu])', r'\\\\', json_str)

        return json.loads(json_str)

    except Exception as e:
        raise ValueError(f"Could not extract valid JSON from response. Reason: {e}")
        print(response)
    

def parse_stream(stream):
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def missing_attributes(attr_dict):
    expected = ["category", "fabric", "color_or_print", "sleeve_length", "occasion", "neckline", "length", "fit", "available_sizes", "price"]
    return [attr for attr in expected if attr not in attr_dict or attr_dict[attr] in [None, "", []]]

def return_attribute_json(user_query):
    user_embedding = model.encode(user_query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(user_embedding, example_embeddings)[0]
    top_indices = cosine_scores.argsort(descending=True)[:3]
    best_index = top_indices[0]
    best_match = list(typical_examples.keys())[best_index]
    confidence = cosine_scores[best_index].item()

    if confidence > 0.6:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query, "user_query": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": json.dumps(typical_examples[best_match]), "attributes": typical_examples[best_match]})
        st.session_state.partial_json = typical_examples[best_match]
    else:
        top_k = {k: typical_examples[k] for i, k in enumerate(typical_examples.keys()) if i in top_indices}
        prior = "\n".join([f"- {k}: {json.dumps(v)}" for k, v in top_k.items()])
        prompt = f"""
Map the user query delimited by triple backticks ```{user_query}``` to structured fashion attributes.
Here are similar examples:
{prior}
Return only JSON with keys:
- category (a single word - tops/pants/etc), fabric (silk/cotton/etc), color_or_print(floral print/charcoal/pastel yellow/etc), sleeve_length (short/long/full/etc), occasion (a single word - party/evening/everyday/etc),
  neckline (v neck/cowl neck/boat neck/etc), length (a single word - Midi/Mini/Short), fit (body hugging/bodycon/relaxed/etc), available_sizes (S/XS/M/etc), price (a single integer)
Return only valid JSON enclosed in curly braces.
Omit any fields that are not known.
Do not include text before or after the JSON.
All string values must be double-quoted.
        """
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": prompt, "user_query": user_query})

        with st.chat_message("assistant"):
            history = [{"role": "user", "content": st.session_state.chat_history[-1]["content"]}]
            stream = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=history,
                stream=True
            )
            response = "".join(parse_stream(stream))
            print(response)
            parsed = extract_json_from_response(response)
            if parsed is None:
                stream = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=history,
                    stream=True
                )
                response = "".join(parse_stream(stream))
                print(response)
                parsed = extract_json_from_response(response)
            

        st.session_state.partial_json = parsed
    missing = missing_attributes(st.session_state.partial_json)
    

    if missing:
        followup_prompt = f"""
We have already extracted these attributes for the user:  
{json.dumps(st.session_state.partial_json, indent=2)}.

The following details are still missing: {', '.join(missing)}.

Write a friendly, single line, short conversational follow-up that encourages the shopper to share the missing info. 
Make it feel like a warm and helpful shopping experience — like a personal stylist assisting them.
Use emojis if appropriate. Output only the follow-up message without double quotes.
        """
        followup_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": followup_prompt}]
        )
        followup = followup_response.choices[0].message.content.strip()

        with st.chat_message("assistant"):
            st.markdown(followup)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": followup,
            "user_query": followup
        })

    else:
        final = st.session_state.partial_json

        scores = keywise_similarity(final, df, model)
        df["cosine_score"] = scores
        # Show top 3 matches
        N=3
        matched_df = df.sort_values(by="cosine_score", ascending=False).head(N)
        display_cols = ["cosine_score"] + [k for k in final.keys() if k in df.columns]
        matched_df = matched_df[display_cols]
        
        # Show result
        if not matched_df.empty:
            st.success(f"Found matching items:")
            st.dataframe(matched_df.head(N))
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": None,  # Skip string content
                "user_query": "Matching items shown below:",
                "dataframe": matched_df.head(N)
            })
        else:
            st.warning("No matching items found.")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": None,  # Skip string content
                "user_query": "No matching items found."
            })

        del st.session_state.partial_json

        next_vibe = "So, any other vibe you want to go for?"
        with st.chat_message("assistant"):
            st.markdown(next_vibe)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": next_vibe,
            "user_query": next_vibe
        })


# UI starts
st.title("👗 Vibe to Fashion Attribute Mapper")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "Hi! Tell me the vibe you're going for, and I’ll help find matching attributes.", "user_query":  "Hi! Tell me the vibe you're going for, and I’ll help find matching attributes."}]

json_pattern = re.compile(r'{.*}', re.DOTALL)

for msg in st.session_state.chat_history:
    content = msg.get("user_query")

    if content:
        st.chat_message(msg["role"]).markdown(content)

    if "dataframe" in msg:
        st.dataframe(msg["dataframe"])

if user_input := st.chat_input("Describe your vibe:"):
    # Detect follow-up response
    if "partial_json" in st.session_state and st.session_state.chat_history[-1]["role"] == "assistant":
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input, "user_query": user_input})

        prompt = f"""
We previously collected these attributes: {json.dumps(st.session_state.partial_json, indent=2)}.
The user added: ```{user_input}```.
Please return a single combined JSON with these keys:
- category (a single word - tops/pants/etc), fabric (silk/cotton/etc), color_or_print(floral print/charcoal/pastel yellow/etc), sleeve_length (short/long/full/etc), occasion (a single word - party/evening/everyday/etc),
  neckline (v neck/cowl neck/boat neck/etc), length (a single word - Midi/Mini/Short), fit (body hugging/bodycon/relaxed/etc), available_sizes (S/XS/M/etc), price (a single integer)
Return only the final JSON.
Return only valid JSON enclosed in curly braces.
Omit any fields that are not known.
Do not include text before or after the JSON.
All string values must be double-quoted.
        """

        st.session_state.chat_history.append({"role": "assistant", "content": prompt})

        with st.chat_message("assistant"):
            history = [{"role": "user", "content": st.session_state.chat_history[-1]["content"]}]
            stream = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=history,
                stream=True
            )
            response = "".join(parse_stream(stream))
            print(response)
            parsed = extract_json_from_response(response)
            if parsed is None:
                stream = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=history,
                    stream=True
                )
                response = "".join(parse_stream(stream))
                print(response)
                parsed = extract_json_from_response(response)

        st.session_state.partial_json.update(parsed)
        final = st.session_state.partial_json
        # st.json(final)

        st.session_state.chat_history.append({"role": "assistant", "content": json.dumps(final, indent=2)})

        scores = keywise_similarity(final, df, model)
        df["cosine_score"] = scores
        N=3
        # Show top 3 matches
        matched_df = df.sort_values(by="cosine_score", ascending=False).head(N)
        display_cols = ["cosine_score"] + [k for k in final.keys() if k in df.columns]
        matched_df = matched_df[display_cols]

        
        # Show result
        if not matched_df.empty:
            st.success(f"Found matching items:")
            st.dataframe(matched_df.head(N))
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": None,  # Skip string content
                "user_query": "Matching items shown below:",
                "dataframe": matched_df.head(N)
            })
        else:
            st.warning("No matching items found.")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": None,  # Skip string content
                "user_query": "No matching items found."
            })

        del st.session_state.partial_json
        next_vibe = "So, any other vibe you want to go for?"
        with st.chat_message("assistant"):
            st.markdown(next_vibe)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": next_vibe,
            "user_query": next_vibe
        })


    else:
        return_attribute_json(user_input)
