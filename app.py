import streamlit as st
import pandas as pd
from transformers import pipeline
import os
import time
import random
from datetime import datetime

# --- App Settings ---
st.set_page_config(page_title="SoulSpace", page_icon="🌿", layout="centered")

# --- Initialize State ---
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to SoulSpace. I'm here to listen. How are you feeling today?"}
    ]
if 'current_mood' not in st.session_state:
    st.session_state.current_mood = "Neutral"

# --- Safety & Logic Data ---
BLOCKED_PHRASES = ["hey guys", "like and subscribe", "what's up", "my name is", "i am an ai", "http", "www"]

FOLLOW_UPS = {
    "Happy": "What's the best part of your day so far?",
    "Sad": "Do you want to talk about what's bringing you down?",
    "Stressed": "What is the most urgent thing on your mind right now?",
    "Anxious": "Let's take a deep breath. Would you like to try a quick grounding exercise?",
    "Motivated": "What is the very first step you can take today?",
    "Neutral": "What's primarily on your agenda for today?",
    "Overwhelmed": "Can you pick just ONE tiny thing to focus on for the next hour?"
}

SAFE_RESPONSES = {
    "Happy": ["It is wonderful to hear you're feeling good!", "Happiness looks great on you."],
    "Sad": ["I am so sorry you are hurting right now.", "Sending gentle thoughts your way."],
    "Stressed": ["Please take a deep breath. Focus on just the very next right step.", "Stress can be so heavy. Remember to be kind to yourself today."],
    "Anxious": ["You are safe right now.", "It's completely normal to feel anxious, but remember this feeling will eventually pass."],
    "Motivated": ["You have everything you need to succeed right now.", "Awesome energy!"],
    "Neutral": ["A calm and peaceful day is a win.", "Taking it one moment at a time is a perfect strategy."],
    "Overwhelmed": ["Everything feels like too much right now, but you only have to do the next small thing.", "Pause. Breathe. You don't have to fix everything right this second."]
}

# Motivation Dictionary
MOTIVATION_MESSAGES = {
    "Sad": "You're stronger than you think. Take it one step at a time.",
    "Stressed": "Take a deep breath. Focus on what you can control right now.",
    "Anxious": "You are safe. This feeling will pass, just breathe.",
    "Happy": "That's wonderful! Keep riding that positive wave.",
    "Motivated": "Let's go! Channel that amazing energy into your goals.",
    "Neutral": "You're doing great. Taking it day by day is enough.",
    "Overwhelmed": "You don't have to do it all at once. Just the very next thing."
}

MOOD_FILE = "mood_history.csv"
if not os.path.exists(MOOD_FILE):
    pd.DataFrame(columns=["Timestamp", "Mood"]).to_csv(MOOD_FILE, index=False)

def log_mood(mood):
    df = pd.read_csv(MOOD_FILE)
    new_entry = pd.DataFrame([{"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Mood": mood}])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(MOOD_FILE, index=False)

# --- NLP Logic Engine ---
@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline("text-generation", model="distilgpt2")

try:
    generator = load_model()
    model_loaded = True
except Exception:
    model_loaded = False

def detect_emotion(text):
    text_lower = text.lower()
    if any(word in text_lower for word in ['sad', 'upset', 'down', 'unhappy', 'depressed', 'cry', 'hurt']):
        return "Sad"
    elif any(word in text_lower for word in ['stress', 'overwhelmed', 'pressure', 'tense', 'exhausted', 'too much']):
        return "Stressed"
    elif any(word in text_lower for word in ['anxious', 'worried', 'nervous', 'panic', 'scared', 'fear']):
        return "Anxious"
    elif any(word in text_lower for word in ['happy', 'good', 'great', 'joy', 'excited', 'amazing']):
        return "Happy"
    elif any(word in text_lower for word in ['motivated', 'productive', 'focused', 'driven', 'ready', 'crush it']):
        return "Motivated"
    return "Neutral"

def clean_ai_text(raw_text, original_prompt):
    if len(raw_text) > len(original_prompt):
        response = raw_text[len(original_prompt):].strip()
        if '"' in response: response = response.split('"')[0]
        
        for char in ['.', '!', '?']:
            if char in response:
                response = response.split(char)[0] + char
                break
        
        response = response.replace('\n', ' ').strip()
        for phrase in BLOCKED_PHRASES:
            if phrase in response.lower(): return ""
                
        if len(response) > 5: return response.capitalize()
    return ""

def generate_response(user_text, current_mood):
    # Overwrite dropdown mood with actively detected textual emotion if it exists
    detected = detect_emotion(user_text)
    active_emotion = detected if detected != "Neutral" else current_mood
    
    # 1. Base Support
    base = random.choice(SAFE_RESPONSES.get(active_emotion, SAFE_RESPONSES["Neutral"]))
    
    # 2. Extract motivational quote
    motivation = MOTIVATION_MESSAGES.get(active_emotion, MOTIVATION_MESSAGES["Neutral"])
    
    # 3. AI Augment (Optional short flavor)
    ai_augment = ""
    if model_loaded:
        prompt = f"User feeling {active_emotion} says: '{user_text}'. Compassionate 1-sentence reply: \""
        try:
            output = generator(prompt, max_new_tokens=15, num_return_sequences=1, temperature=0.7, top_p=0.9, repetition_penalty=1.5, do_sample=True, pad_token_id=50256)
            cleaned = clean_ai_text(output[0]['generated_text'], prompt)
            if cleaned: ai_augment = cleaned
        except:
             pass 
             
    # Strict structure enforcing 1-2 sentence reliable responses
    if ai_augment:
        final_response = f"{ai_augment}\n\n*{motivation}*"
    else:
        final_response = f"{base}\n\n*{motivation}*"
        
    return final_response.strip()

# --- Sidebar ---
with st.sidebar:
    st.title("SoulSpace 🌿")
    st.caption("A peaceful space for your thoughts.")
    st.divider()
    
    st.subheader("🧭 Mood Tracker")
    moods = ["Happy", "Sad", "Stressed", "Anxious", "Motivated", "Neutral", "Overwhelmed"]
    selected_mood = st.selectbox("How are you feeling?", moods, index=moods.index(st.session_state.current_mood))
    
    if selected_mood != st.session_state.current_mood:
        st.session_state.current_mood = selected_mood
        log_mood(selected_mood)
        
        # Base reaction using the exact motivational logic when they just switch dropdowns
        motivation = MOTIVATION_MESSAGES.get(selected_mood, MOTIVATION_MESSAGES["Neutral"])
        bot_reaction = f"I noticed you're feeling {selected_mood}. \n\n*{motivation}*"
        st.session_state.messages.append({"role": "assistant", "content": bot_reaction})
        if hasattr(st, "rerun"): st.rerun()
        else: st.experimental_rerun()

    st.divider()
    
    st.subheader("📊 Recent Trends")
    try:
        df = pd.read_csv(MOOD_FILE)
        if not df.empty:
            recent_df = df.tail(15) 
            mood_counts = recent_df['Mood'].value_counts()
            st.bar_chart(mood_counts)
        else:
            st.info("Log your mood to see trends.")
    except:
        pass

# --- Main App Body ---
st.title("SoulSpace 🌿")
st.subheader("Your private mental wellness companion")
st.divider()

# Rendering standard Streamlit Chat
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🌿"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# User Input processing
if prompt_text := st.chat_input("Write whatever is on your mind..."):
    # Render user immediately
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user", avatar="👤"):
        st.write(prompt_text)
    
    # Assistant response thinking placeholder
    with st.chat_message("assistant", avatar="🌿"):
        with st.spinner("Thinking..."):
            time.sleep(1.0) # Smooth delay
            final_ans = generate_response(prompt_text, st.session_state.current_mood)
        st.write(final_ans)
        st.session_state.messages.append({"role": "assistant", "content": final_ans})

st.divider()

# Journal Section using standard Streamlit expander
with st.expander("📓 Guided Wellness Journal"):
    st.write("Take a moment to reflect.")
    
    prompts_map = {
        "Happy": "What exactly made you smile today?",
        "Sad": "What is weighing on your heart right now?",
        "Stressed": "Action plan: Brain dump everything on your mind, then pick ONE to focus on.",
        "Anxious": "Let's ground ourselves. What are you worried about?",
        "Motivated": "Write down your #1 goal right now. Break it down into 3 tiny steps.",
        "Neutral": "What is one incredibly small, mundane thing you appreciate today?",
        "Overwhelmed": "Write down ONE single thing you have absolute control over right now."
    }
    
    st.info(f"**Prompt:** {prompts_map.get(st.session_state.current_mood, 'Write whatever is on your mind.')}")
    
    journal_text = st.text_area("Your entries are safely stored for this session:", height=150, placeholder="Begin journaling here...")
    
    if st.button("Save Entry", use_container_width=True):
        if 'journal_entries' not in st.session_state:
            st.session_state.journal_entries = []
            
        if journal_text.strip():
            st.session_state.journal_entries.append({
                "time": datetime.now().strftime("%I:%M %p"),
                "mood": st.session_state.current_mood,
                "text": journal_text.strip()
            })
            st.success("Journal entry saved!")
        else:
            st.warning("Please type your thoughts before saving.")
            
    if 'journal_entries' in st.session_state and st.session_state.journal_entries:
        st.markdown("---")
        st.markdown("#### Today's Reflections")
        for i, entry in enumerate(reversed(st.session_state.journal_entries)):
            if i >= 3: break
            st.caption(f"⏰ {entry['time']} · {entry['mood']}")
            st.write(f"*{entry['text']}*")
