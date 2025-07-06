import streamlit as st

def render_audio_input_if_visible():
    """
    Renders the audio input widget and a close button if st.session_state['show_mic'] is True.
    Returns the audio file if recorded, else None.
    """
    audio_file = None
    if st.session_state.get('show_mic', False):
        audio_file = st.audio_input("Record your query", key="mic_input", label_visibility="collapsed")
        if audio_file:
            st.success("Your query has been received! Please scroll down to view the results.")
        if st.button("Close", key="close_mic_input"):
            st.session_state['show_mic'] = False
    return audio_file 