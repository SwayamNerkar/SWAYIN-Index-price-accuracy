import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="SWAYIN.AI | Smart Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
#  LANDING PAGE & AUTHENTICATION PROTOCOL
# ══════════════════════════════════════════════════════════════════════
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "landing"

if not st.session_state.logged_in:
    # ── Landing Page Specialized CSS ──
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        /* Force-hide Streamlit Chrome & Sidebar on Landing Page */
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="collapsedControl"] { display: none !important; }
        header { visibility: hidden !important; height: 0 !important; }
        .stApp { background-color: transparent !important; }
        
        /* Remove Default Streamlit Padding */
        .block-container { 
            padding-top: 0 !important; 
            padding-bottom: 0 !important; 
            padding-left: 0 !important; 
            padding-right: 0 !important; 
            max-width: 100% !important; 
        }

        /* ── Hero Section ── */
        .landing-hero {
            position: relative;
            height: 60vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            padding: 0 20px;
            z-index: 10;
        }
        .hero-title {
            font-family: 'Inter', sans-serif;
            font-size: 6.5rem;
            font-weight: 800;
            color: #f8fafc;
            letter-spacing: -0.04em;
            margin-bottom: 15px;
            z-index: 10;
            text-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }
        .hero-title span { color: #f59e0b; /* Amber */ }
        .hero-subtitle {
            font-family: 'Inter', sans-serif;
            font-size: 1.4rem;
            color: #cbd5e1;
            max-width: 700px;
            line-height: 1.6;
            z-index: 10;
            text-shadow: 0 4px 10px rgba(0,0,0,0.5);
        }

        /* ── Buttons Style via CSS ── */
        div[data-testid="stButton"] button {
            border-radius: 12px !important; 
            font-weight: 800 !important;
            height: 54px !important;
            font-size: 1.1rem !important; 
            text-transform: uppercase; 
            letter-spacing: 0.08em;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            border: none !important;
        }
        /* Login button - Outline/Ghost style */
        .btn-login div[data-testid="stButton"] button {
            background: rgba(11, 15, 26, 0.6) !important;
            color: #f59e0b !important;
            border: 2px solid #f59e0b !important;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }
        .btn-login div[data-testid="stButton"] button:hover {
            background: rgba(245, 158, 11, 0.15) !important;
            transform: translateY(-3px) scale(1.02); 
            box-shadow: 0 15px 30px rgba(245,158,11,0.2) !important;
        }
        
        /* Signup button - Solid style */
        .btn-signup div[data-testid="stButton"] button {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
            color: #0b0f1a !important;
            box-shadow: 0 10px 25px rgba(245,158,11,0.3) !important;
        }
        .btn-signup div[data-testid="stButton"] button:hover {
            transform: translateY(-3px) scale(1.02); 
            box-shadow: 0 20px 40px rgba(245,158,11,0.5) !important;
        }

        /* ── Parallax Feature Section ── */
        .feature-panel {
            background: linear-gradient(rgba(11,15,26,0.95), rgba(11,15,26,0.99));
            padding: 100px 40px;
            text-align: center;
            border-top: 1px solid rgba(245,158,11,0.15);
            position: relative;
            z-index: 10;
        }
        .feat-title {
            font-family: 'Inter', sans-serif;
            font-size: 2.5rem; color: #f8fafc; font-weight: 800; margin-bottom: 60px;
            letter-spacing: -0.02em;
        }
        .feat-grid { display: flex; gap: 30px; justify-content: center; flex-wrap: wrap; }
        .feat-box {
            background: rgba(30,41,59,0.5); 
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255,255,255,0.05); 
            border-radius: 20px;
            padding: 40px 35px; 
            width: 350px; 
            text-align: left;
            transition: transform 0.4s ease, border-color 0.4s ease, box-shadow 0.4s ease;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }
        .feat-box:hover {
            transform: translateY(-10px);
            border-color: rgba(245,158,11,0.5);
            box-shadow: 0 20px 40px rgba(245,158,11,0.15);
        }
        .feat-box h3 { 
            color: #f59e0b; font-size: 1.4rem; font-family: 'Inter', sans-serif; 
            font-weight: 800; margin-bottom: 15px; margin-top: 0;
            letter-spacing: -0.02em;
        }
        .feat-box p { 
            color: #94a3b8; font-size: 1rem; line-height: 1.7; font-family: 'Inter', sans-serif; margin: 0;
        }
        
        /* Clean up Auth Form styles & Animations */
        @keyframes floatUp {
            0% { opacity: 0; transform: translateY(40px) scale(0.95); }
            100% { opacity: 1; transform: translateY(0) scale(1); }
        }
        .auth-wrapper {
            max-width: 480px; margin: 40px auto 60px auto; 
            padding: 45px 40px;
            background: rgba(17,24,39,0.85);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border: 1px solid rgba(245,158,11,0.2); 
            border-radius: 24px;
            box-shadow: 0 30px 60px -12px rgba(0,0,0,0.8), 0 0 40px rgba(245,158,11,0.1);
            position: relative; z-index: 20;
            animation: floatUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        .auth-header {
            text-align: center;
            margin-bottom: 30px;
        }
        .auth-header h2 {
            font-family: 'Inter', sans-serif;
            color: #f8fafc;
            font-size: 2rem;
            font-weight: 800;
            margin: 0 0 10px 0;
        }
        .auth-header p {
            color: #94a3b8;
            font-size: 1rem;
            margin: 0;
        }
        .stTextInput input { 
            border-radius: 10px !important; padding: 14px 16px !important; 
            font-size: 1.05rem !important; background: rgba(0,0,0,0.4) !important; 
            border: 1px solid rgba(255,255,255,0.1) !important;
            color: #f8fafc !important;
        }
        .stTextInput input:focus {
            border-color: #f59e0b !important;
            box-shadow: 0 0 0 1px #f59e0b !important;
        }

        /* ── Smooth Scrolling Marquee ── */
        .marquee-container {
            width: 100%; overflow: hidden; background: #f59e0b; color: #0b0f1a;
            padding: 15px 0; font-family: 'JetBrains Mono', monospace; font-weight: 800; 
            font-size: 0.9rem; letter-spacing: 0.15em; text-transform: uppercase;
            position: relative;
            z-index: 10;
        }
        .marquee-content {
            display: inline-block; white-space: nowrap; animation: marqueeScroll 25s linear infinite;
        }
        @keyframes marqueeScroll {
            0% { transform: translateX(0); } 100% { transform: translateX(-50%); }
        }
        
        /* Back button */
        .btn-back div[data-testid="stButton"] button {
            background: transparent !important;
            color: #94a3b8 !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            box-shadow: none !important;
            height: 44px !important;
            font-size: 0.9rem !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        .btn-back div[data-testid="stButton"] button:hover {
            color: #f8fafc !important;
            border-color: rgba(255,255,255,0.3) !important;
            transform: translateY(-2px) !important;
            background: rgba(255,255,255,0.05) !important;
        }
        
        /* Remove background from tabs content if any */
        .stTabs [data-baseweb="tab-list"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Inject WebGL Background ──
    st.markdown("<div id='vanta-bg' style='position:fixed; top:0; left:0; width:100vw; height:100vh; z-index:0;'></div>", unsafe_allow_html=True)
    import streamlit.components.v1 as components
    components.html("""
        <script>
            const parentDoc = window.parent.document;
            if (!parentDoc.getElementById('vanta-script')) {
                const threeScript = parentDoc.createElement('script');
                threeScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js';
                threeScript.id = 'three-script';
                parentDoc.head.appendChild(threeScript);
                
                threeScript.onload = function() {
                    const vantaScript = parentDoc.createElement('script');
                    vantaScript.src = 'https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.net.min.js';
                    vantaScript.id = 'vanta-script';
                    parentDoc.head.appendChild(vantaScript);
                    
                    vantaScript.onload = function() {
                        const bgElement = parentDoc.getElementById('vanta-bg');
                        if (bgElement) {
                            window.parent.VANTA.NET({
                              el: bgElement,
                              mouseControls: true,
                              touchControls: true,
                              gyroControls: false,
                              minHeight: 200.00,
                              minWidth: 200.00,
                              scale: 1.00,
                              scaleMobile: 1.00,
                              color: 0xf59e0b,
                              backgroundColor: 0x0b0f1a,
                              points: 14.00,
                              maxDistance: 22.00,
                              spacing: 16.00,
                              showDots: true
                            });
                        }
                    }
                }
            } else {
                // If script exists, re-init in case element was removed and re-added
                setTimeout(() => {
                    if (window.parent.VANTA && window.parent.VANTA.NET) {
                        const bgElement = parentDoc.getElementById('vanta-bg');
                        // Ensure old canvas is removed to prevent duplicates
                        if (bgElement && bgElement.children.length === 0) {
                            window.parent.VANTA.NET({
                                el: bgElement,
                                color: 0xf59e0b,
                                backgroundColor: 0x0b0f1a,
                                points: 14.00,
                                maxDistance: 22.00,
                                spacing: 16.00,
                                showDots: true
                            });
                        }
                    }
                }, 500);
            }
        </script>
    """, height=0, width=0)

    if st.session_state.auth_mode == "landing":
        # 1. Hero Section
        st.markdown("""
            <div class="landing-hero">
                <div class="hero-title">SWAYIN<span>.AI</span></div>
                <div class="hero-subtitle">The next-generation terminal for algorithmic predictions, sentiment intelligence, and technical confluence.</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Action Buttons
        st.write("") # Spacer
        col_space1, col_login, col_space2, col_signup, col_space3 = st.columns([1.5, 2, 0.2, 2, 1.5])
        
        with col_login:
            st.markdown('<div class="btn-login">', unsafe_allow_html=True)
            if st.button("SECURE LOGIN", use_container_width=True):
                st.session_state.auth_mode = "login"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_signup:
            st.markdown('<div class="btn-signup">', unsafe_allow_html=True)
            if st.button("CREATE ACCOUNT", use_container_width=True):
                st.session_state.auth_mode = "signup"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('<div style="height: 15vh;"></div>', unsafe_allow_html=True)

    elif st.session_state.auth_mode == "login":
        # Back Button
        st.markdown('<div style="position: absolute; z-index: 30; top: 0px; left: 0px;" class="btn-back">', unsafe_allow_html=True)
        if st.button("← Back to Home"):
            st.session_state.auth_mode = "landing"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Login Form
        st.markdown("""
            <div class="auth-wrapper">
                <div class="auth-header">
                    <h2>Welcome Back</h2>
                    <p>Enter your credentials to access the terminal</p>
                </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            login_email = st.text_input("Terminal ID / Email Address", placeholder="e.g. trader@swayin.ai")
            login_pass = st.text_input("Access Key / Password", type="password")
            st.markdown('<div class="btn-signup">', unsafe_allow_html=True)
            if st.form_submit_button("AUTHORIZE ACCESS", use_container_width=True):
                if login_email and login_pass:
                    import auth_system
                    success, msg = auth_system.sign_in(login_email, login_pass)
                    if success:
                        st.session_state.logged_in = True
                        st.rerun()
                    else:
                        st.error(f"Access Denied: {msg}")
                else:
                    st.error("Please provide both email and password.")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.auth_mode == "signup":
        # Back Button
        st.markdown('<div style="position: absolute; z-index: 30; top: 0px; left: 0px;" class="btn-back">', unsafe_allow_html=True)
        if st.button("← Back to Home"):
            st.session_state.auth_mode = "landing"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Signup Form
        st.markdown("""
            <div class="auth-wrapper">
                <div class="auth-header">
                    <h2>Deploy Terminal</h2>
                    <p>Register to unlock AI predictive capabilities</p>
                </div>
        """, unsafe_allow_html=True)
        
        with st.form("signup_form"):
            signup_name = st.text_input("Full Name", placeholder="e.g. John Doe")
            signup_email = st.text_input("Email Address", placeholder="e.g. trader@swayin.ai")
            signup_pass = st.text_input("Create Access Key", type="password", placeholder="Strong password")
            st.markdown('<div class="btn-signup">', unsafe_allow_html=True)
            if st.form_submit_button("CREATE ACCOUNT", use_container_width=True):
                if signup_email and signup_pass and signup_name:
                    import auth_system
                    success, msg = auth_system.sign_up(signup_email, signup_pass, signup_name)
                    if success:
                        st.success("Registration successful! Proceed to Secure Login.")
                        st.session_state.auth_mode = "login"
                    else:
                        st.error(f"Failed to Deploy: {msg}")
                else:
                    st.error("All fields are required.")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.auth_mode == "landing":
        # 3. Scrolling Animated Marquee
        marquee_text = "&nbsp; • &nbsp; DEEP LSTM PREDICTIONS &nbsp; • &nbsp; LIVE MARKET SENTIMENT &nbsp; • &nbsp; TECHNICAL CONFLUENCE &nbsp; • &nbsp; EXPLAINABLE AI &nbsp; • &nbsp; SECURE TERMINAL &nbsp; • &nbsp; NEXT GEN ALGORITHMS "
        st.markdown(f"""
            <div class="marquee-container">
                <div class="marquee-content">
                    {marquee_text * 4}
                </div>
            </div>
        """, unsafe_allow_html=True)

        # 4. Features Parallax Sub-section
        st.markdown("""
            <div class="feature-panel">
                <div class="feat-title">Why Professionals Choose SWAYIN.AI</div>
                <div class="feat-grid">
                    <div class="feat-box">
                        <h3>Deep Neural Processing</h3>
                        <p>Industry-grade multi-layer LSTM architecture designed to capture volatile market sequences and time-series anomalies.</p>
                    </div>
                    <div class="feat-box">
                        <h3>Transparent XAI</h3>
                        <p>Don't just get predictions. Understand exactly which technical signals triggered the AI output via SHAP-driven analytics.</p>
                    </div>
                    <div class="feat-box">
                        <h3>Live Sentiment Edge</h3>
                        <p>Real-time NLP sentiment extraction from top financial news headlines, actively combining with quantitative indicators.</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # 5. Footer
        st.markdown("""
            <div style="text-align:center; padding:30px; background:#0b0f1a; color:#475569; font-size:0.9rem; font-family:'Inter',sans-serif; position:relative; z-index:10; border-top: 1px solid rgba(255,255,255,0.05);">
                &copy; 2026 SWAYIN.AI Intelligence Systems. Not actual financial advice.
            </div>
        """, unsafe_allow_html=True)

    st.stop()  # END LANDING PAGE EXECUTION

# ══════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM — Professional Navy + Amber Financial Terminal Theme
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Root Variables ── */
    :root {
        --navy-900: #0b0f1a;
        --navy-800: #111827;
        --navy-700: #1a2235;
        --navy-600: #1e2d45;
        --navy-500: #253550;
        --amber:    #f59e0b;
        --amber-dim: #d97706;
        --amber-glow: rgba(245, 158, 11, 0.15);
        --emerald: #10b981;
        --rose:    #f43f5e;
        --sky:     #38bdf8;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted:    #64748b;
        --border:  rgba(248, 250, 252, 0.07);
        --card-bg: rgba(17, 24, 39, 0.75);
        --success-bg: rgba(16, 185, 129, 0.1);
        --danger-bg:  rgba(244, 63, 94, 0.1);
    }

    /* ── Global Reset ── */
    html, body, .stApp {
        background: var(--navy-900) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ── Top Navbar ── */
    .swayin-navbar {
        position: sticky;
        top: 0;
        z-index: 999;
        background: rgba(11, 15, 26, 0.95);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-bottom: 1px solid var(--border);
        padding: 0 32px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        height: 64px;
        margin: -1rem -1rem 2rem -1rem;
    }
    .navbar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .navbar-logo {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, var(--amber) 0%, #fbbf24 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        font-weight: 800;
        color: #0b0f1a;
        letter-spacing: -0.05em;
        box-shadow: 0 4px 14px rgba(245,158,11,0.3);
        flex-shrink: 0;
    }
    .navbar-name {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }
    .navbar-name span {
        color: var(--amber);
    }
    .navbar-tagline {
        font-size: 0.7rem;
        color: var(--text-muted);
        font-weight: 400;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 1px;
    }
    .navbar-pills {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .nav-pill {
        background: var(--navy-700);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--text-secondary);
        letter-spacing: 0.03em;
    }
    .nav-pill.active {
        background: var(--amber-glow);
        border-color: rgba(245,158,11,0.35);
        color: var(--amber);
    }
    .nav-status {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.72rem;
        color: var(--emerald);
        font-weight: 500;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--emerald);
        animation: pulse-dot 2s infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%       { opacity: 0.5; transform: scale(0.8); }
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--navy-800) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }
    .sidebar-section-title {
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1.5rem 0 0.75rem 0;
        padding-left: 2px;
    }

    /* ── Cards ── */
    .card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 16px;
        backdrop-filter: blur(12px);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .card:hover {
        border-color: rgba(245,158,11,0.2);
        box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    }

    /* ── Section Headers ── */
    .section-label {
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--amber);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 4px;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border);
        margin-left: 4px;
    }

    /* ── Metric cards ── */
    div[data-testid="metric-container"] {
        background: var(--card-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        padding: 18px 20px !important;
        transition: all 0.25s ease !important;
        position: relative;
        overflow: hidden;
    }
    div[data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        background: linear-gradient(180deg, var(--amber), transparent);
        border-radius: 3px 0 0 3px;
    }
    div[data-testid="metric-container"]:hover {
        border-color: rgba(245,158,11,0.25) !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(0,0,0,0.4);
    }
    div[data-testid="metric-container"] label {
        color: var(--text-muted) !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        font-family: 'Inter', sans-serif !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: -0.02em !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }

    /* ── Signal Badges ── */
    .signal-buy {
        background: var(--success-bg);
        border: 1.5px solid rgba(16,185,129,0.4);
        color: var(--emerald);
        font-size: 1rem;
        font-weight: 700;
        padding: 12px 28px;
        border-radius: 12px;
        text-align: center;
        letter-spacing: 0.1em;
        font-family: 'JetBrains Mono', monospace;
        display: inline-block;
        width: 100%;
    }
    .signal-sell {
        background: var(--danger-bg);
        border: 1.5px solid rgba(244,63,94,0.4);
        color: var(--rose);
        font-size: 1rem;
        font-weight: 700;
        padding: 12px 28px;
        border-radius: 12px;
        text-align: center;
        letter-spacing: 0.1em;
        font-family: 'JetBrains Mono', monospace;
        display: inline-block;
        width: 100%;
    }

    /* ── Sentiment Block ── */
    .sentiment-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 24px;
        text-align: center;
    }
    .sentiment-card .s-label {
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }
    .sentiment-card .s-value {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        margin: 6px 0 4px 0;
    }
    .s-bullish { color: var(--emerald); }
    .s-bearish { color: var(--rose); }
    .s-neutral  { color: var(--text-secondary); }
    .sentiment-card .s-score {
        font-size: 0.78rem;
        color: var(--text-muted);
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Driver card ── */
    .driver-card {
        background: var(--navy-700);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .driver-card .d-title { font-weight: 600; font-size: 0.85rem; color: var(--text-primary); }
    .driver-card .d-sub   { font-size: 0.78rem; color: var(--text-muted); margin-top: 4px; }

    /* ── Risk Row ── */
    .risk-row {
        display: flex;
        gap: 12px;
        margin-top: 12px;
    }
    .risk-chip {
        flex: 1;
        background: var(--navy-700);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 8px 12px;
        text-align: center;
        font-size: 0.72rem;
    }
    .risk-chip .rc-label { color: var(--text-muted); margin-bottom: 2px; font-weight: 500; }
    .risk-chip .rc-value { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.88rem; }
    .rc-sl { color: var(--rose); }
    .rc-tp { color: var(--emerald); }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--navy-800) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 2px !important;
        border: 1px solid var(--border) !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        color: var(--text-muted) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 8px 18px !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--amber-glow) !important;
        color: var(--amber) !important;
        font-weight: 600 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--amber) 0%, var(--amber-dim) 100%) !important;
        color: #0b0f1a !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.04em !important;
        padding: 10px 0 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(245,158,11,0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(245,158,11,0.4) !important;
    }

    /* ── Selectbox / Radio / Inputs ── */
    .stSelectbox [data-baseweb="select"] > div,
    .stRadio > div {
        background: var(--navy-700) !important;
        border-color: var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }

    /* ── Dataframe ── */
    .stDataFrame {
        background: var(--card-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* ── Divider ── */
    hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

    /* ── Caption override ── */
    .stCaption, small { color: var(--text-muted) !important; font-size: 0.78rem !important; }

    /* ── Info box ── */
    .stAlert { border-radius: 12px !important; border-left-width: 3px !important; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer { visibility: hidden; }
    header { background: transparent !important; }

    /* ── Hide raw-HTML code tooltip / source-view overlay ── */
    div[data-testid="stCode"],
    .stCode,
    div[data-testid="stMarkdownContainer"] pre,
    div[data-testid="stMarkdownContainer"] code { display: none !important; }

    /* ── Ensure the navbar markdown wrapper has no extra padding ── */
    div[data-testid="stMarkdownContainer"]:has(.swayin-navbar) {
        margin: 0 !important;
        padding: 0 !important;
    }

    /* ── Global index market ticker ── */
    .global-ticker {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        background: var(--navy-700);
        border: 1px solid var(--border);
        border-radius: 8px;
        font-size: 0.72rem;
        white-space: nowrap;
    }
    .gt-name { color: var(--text-muted); font-weight: 500; }
    .gt-up   { color: var(--emerald); font-weight: 600; font-family: 'JetBrains Mono', monospace; }
    .gt-dn   { color: var(--rose);    font-weight: 600; font-family: 'JetBrains Mono', monospace; }

    /* ── Welcome Hero ── */
    .hero-wrap {
        text-align: center;
        padding: 70px 20px 80px;
    }
    .hero-badge {
        display: inline-block;
        background: var(--amber-glow);
        border: 1px solid rgba(245,158,11,0.3);
        color: var(--amber);
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        padding: 5px 14px;
        border-radius: 99px;
        margin-bottom: 20px;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: var(--text-primary);
        line-height: 1.1;
        margin-bottom: 10px;
    }
    .hero-title span { color: var(--amber); }
    .hero-subtitle {
        font-size: 1.05rem;
        color: var(--text-secondary);
        font-weight: 400;
        margin-bottom: 40px;
        line-height: 1.6;
    }
    .hero-hint {
        font-size: 0.82rem;
        color: var(--text-muted);
        margin-top: 12px;
    }
    .hero-features {
        display: flex;
        justify-content: center;
        gap: 32px;
        flex-wrap: wrap;
        margin-top: 40px;
    }
    .hero-feat {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
    }
    .hero-feat-icon {
        font-size: 1.5rem;
        width: 52px;
        height: 52px;
        background: var(--navy-700);
        border: 1px solid var(--border);
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .hero-feat-label {
        font-size: 0.72rem;
        color: var(--text-muted);
        font-weight: 500;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  DATA HELPERS
#
@st.cache_data(show_spinner="Fetching market data…", ttl=300)
def _get_data(symbol, interval, period):
    from data_fetcher import fetch_stock_data
    from features     import engineer_features
    df_raw = fetch_stock_data(symbol=symbol, interval=interval, period=period)
    df     = engineer_features(df_raw)
    return df


@st.cache_resource(show_spinner="Running LSTM Intelligence Engine…")
def _get_model_and_preds(symbol, interval, period, retrain):
    from model      import preprocess_data, build_model, train_model, save_model, load_model
    from predictor  import predict_sets, predict_next_day, predict_next_5min, inverse_actual
    from utils      import evaluate_model, backtest
    from config     import MODEL_PATH, TIME_STEP
    from data_fetcher import fetch_news_sentiment
    from xai_engine   import get_feature_importance

    df = _get_data(symbol, interval, period)
    X_train, y_train, X_test, y_test, scaler, train_size, target_idx, numeric_df = \
        preprocess_data(df)
    n_features  = X_train.shape[2]
    input_shape = (TIME_STEP, n_features)

    model_loaded = False
    if os.path.exists(MODEL_PATH) and not retrain:
        try:
            model        = load_model(MODEL_PATH)
            history      = None
            model_loaded = True
        except Exception as e:
            st.error(f"Model load error: {e}")
            model_loaded = False

    if not model_loaded:
        model   = build_model(input_shape)
        history = train_model(model, X_train, y_train)
        save_model(model)

    train_pred, test_pred = predict_sets(model, X_train, X_test, scaler, target_idx, n_features)
    actual_train, actual_test = inverse_actual(y_train, y_test, scaler, target_idx, n_features)

    current_price = float(df["Close"].iloc[-1])
    scaled_all    = scaler.transform(numeric_df)
    next_day  = predict_next_day(model, scaled_all, scaler, target_idx, n_features, current_price)
    next_5min = predict_next_5min(model, scaled_all, scaler, target_idx, n_features, current_price)

    sentiment_data = fetch_news_sentiment(symbol)
    last_seq       = scaled_all[-TIME_STEP:].reshape(1, TIME_STEP, n_features)
    feature_names  = numeric_df.columns.tolist()
    xai_results    = get_feature_importance(model, last_seq, feature_names)

    metrics   = evaluate_model(actual_test, test_pred)
    bt_result = backtest(actual_test, test_pred)

    return {
        "df"           : df,
        "train_pred"   : train_pred,
        "test_pred"    : test_pred,
        "actual_train" : actual_train,
        "actual_test"  : actual_test,
        "history"      : history,
        "current_price": current_price,
        "next_day"     : next_day,
        "next_5min"    : next_5min,
        "metrics"      : metrics,
        "bt_result"    : bt_result,
        "train_size"   : train_size,
        "sentiment"    : sentiment_data,
        "xai"          : xai_results,
    }


# ══════════════════════════════════════════════════════════════════════
#  MARKET DATA

INDIAN_MARKET = {
    "📊 Indices": [
        ("NIFTY 50",            "^NSEI"),
        ("SENSEX (BSE 30)",     "^BSESN"),
        ("NIFTY Bank",          "^NSEBANK"),
        ("NIFTY IT",            "^CNXIT"),
        ("NIFTY Midcap 100",    "^NSMIDCP100"),
        ("NIFTY Smallcap 100",  "NIFTY_SMLCAP100.NS"),
        ("NIFTY FMCG",         "^CNXFMCG"),
        ("NIFTY Auto",         "^CNXAUTO"),
        ("NIFTY Pharma",       "^CNXPHARMA"),
        ("NIFTY Metal",        "^CNXMETAL"),
        ("NIFTY Energy",       "^CNXENERGY"),
        ("NIFTY Infra",        "^CNXINFRA"),
        ("NIFTY Realty",       "^CNXREALTY"),
        ("NIFTY PSU Bank",     "^CNXPSUBANK"),
        ("NIFTY Media",        "^CNXMEDIA"),
    ],
    "🏦 Banking & Finance": [
        ("HDFC Bank",          "HDFCBANK.NS"),
        ("ICICI Bank",         "ICICIBANK.NS"),
        ("State Bank of India","SBIN.NS"),
        ("Kotak Mahindra Bank","KOTAKBANK.NS"),
        ("Axis Bank",          "AXISBANK.NS"),
        ("Bank of Baroda",     "BANKBARODA.NS"),
        ("Punjab National Bank","PNB.NS"),
        ("Canara Bank",        "CANBK.NS"),
        ("IndusInd Bank",      "INDUSINDBK.NS"),
        ("Federal Bank",       "FEDERALBNK.NS"),
        ("Bajaj Finance",      "BAJFINANCE.NS"),
        ("Bajaj Finserv",      "BAJAJFINSV.NS"),
        ("HDFC Life",          "HDFCLIFE.NS"),
        ("SBI Life Insurance", "SBILIFE.NS"),
        ("LIC Housing Finance","LICHSGFIN.NS"),
        ("Muthoot Finance",    "MUTHOOTFIN.NS"),
        ("Shriram Finance",    "SHRIRAMFIN.NS"),
    ],
    "💻 Information Technology": [
        ("TCS",                "TCS.NS"),
        ("Infosys",            "INFY.NS"),
        ("HCL Technologies",   "HCLTECH.NS"),
        ("Wipro",              "WIPRO.NS"),
        ("Tech Mahindra",      "TECHM.NS"),
        ("LTIMindtree",        "LTIM.NS"),
        ("Mphasis",            "MPHASIS.NS"),
        ("Coforge",            "COFORGE.NS"),
        ("Persistent Systems", "PERSISTENT.NS"),
        ("Oracle Financial",   "OFSS.NS"),
        ("Tata Elxsi",         "TATAELXSI.NS"),
    ],
    "🏭 Large Cap — Diversified": [
        ("Reliance Industries","RELIANCE.NS"),
        ("TCS",                "TCS.NS"),
        ("HDFC Bank",          "HDFCBANK.NS"),
        ("Infosys",            "INFY.NS"),
        ("ITC",                "ITC.NS"),
        ("Larsen & Toubro",    "LT.NS"),
        ("Hindustan Unilever", "HINDUNILVR.NS"),
        ("Asian Paints",       "ASIANPAINT.NS"),
        ("Nestle India",       "NESTLEIND.NS"),
        ("Maruti Suzuki",      "MARUTI.NS"),
        ("Titan Company",      "TITAN.NS"),
        ("Adani Enterprises",  "ADANIENT.NS"),
        ("Adani Ports",        "ADANIPORTS.NS"),
        ("Power Grid Corp",    "POWERGRID.NS"),
        ("NTPC",               "NTPC.NS"),
        ("Coal India",         "COALINDIA.NS"),
        ("ONGC",               "ONGC.NS"),
        ("Tata Steel",         "TATASTEEL.NS"),
        ("JSW Steel",          "JSWSTEEL.NS"),
        ("UltraTech Cement",   "ULTRACEMCO.NS"),
    ],
    "🚗 Automobile": [
        ("Maruti Suzuki",      "MARUTI.NS"),
        ("Tata Motors",        "TATAMOTORS.NS"),
        ("Mahindra & Mahindra","M&M.NS"),
        ("Hero MotoCorp",      "HEROMOTOCO.NS"),
        ("Bajaj Auto",         "BAJAJ-AUTO.NS"),
        ("Eicher Motors",      "EICHERMOT.NS"),
        ("TVS Motor",          "TVSMOTOR.NS"),
        ("Bosch India",        "BOSCHLTD.NS"),
        ("Ashok Leyland",      "ASHOKLEY.NS"),
        ("MRF",                "MRF.NS"),
        ("Apollo Tyres",       "APOLLOTYRE.NS"),
    ],
    "💊 Pharma & Healthcare": [
        ("Sun Pharma",         "SUNPHARMA.NS"),
        ("Dr Reddy's",         "DRREDDY.NS"),
        ("Cipla",              "CIPLA.NS"),
        ("Divi's Labs",        "DIVISLAB.NS"),
        ("Aurobindo Pharma",   "AUROPHARMA.NS"),
        ("Biocon",             "BIOCON.NS"),
        ("Lupin",              "LUPIN.NS"),
        ("Abbott India",       "ABBOTINDIA.NS"),
        ("Alkem Labs",         "ALKEM.NS"),
        ("Torrent Pharma",     "TORNTPHARM.NS"),
        ("Apollo Hospitals",   "APOLLOHOSP.NS"),
        ("Max Healthcare",     "MAXHEALTH.NS"),
        ("Fortis Healthcare",  "FORTIS.NS"),
    ],
    "🛒 FMCG & Consumer": [
        ("Hindustan Unilever", "HINDUNILVR.NS"),
        ("ITC",                "ITC.NS"),
        ("Nestle India",       "NESTLEIND.NS"),
        ("Britannia",          "BRITANNIA.NS"),
        ("Dabur India",        "DABUR.NS"),
        ("Marico",             "MARICO.NS"),
        ("Godrej Consumer",    "GODREJCP.NS"),
        ("Colgate-Palmolive",  "COLPAL.NS"),
        ("United Breweries",   "UBL.NS"),
        ("Radico Khaitan",     "RADICO.NS"),
        ("Emami",              "EMAMILTD.NS"),
    ],
    "⚡ Energy & Oil/Gas": [
        ("ONGC",               "ONGC.NS"),
        ("Reliance Industries","RELIANCE.NS"),
        ("BPCL",               "BPCL.NS"),
        ("HPCL",               "HPCL.NS"),
        ("Indian Oil Corp",    "IOC.NS"),
        ("GAIL India",         "GAIL.NS"),
        ("Petronet LNG",       "PETRONET.NS"),
        ("Adani Green Energy", "ADANIGREEN.NS"),
        ("Adani Total Gas",    "ATGL.NS"),
        ("Tata Power",         "TATAPOWER.NS"),
        ("Power Grid Corp",    "POWERGRID.NS"),
        ("NTPC",               "NTPC.NS"),
    ],
    "🏗️ Infrastructure & Metals": [
        ("Larsen & Toubro",    "LT.NS"),
        ("Tata Steel",         "TATASTEEL.NS"),
        ("JSW Steel",          "JSWSTEEL.NS"),
        ("Hindalco",           "HINDALCO.NS"),
        ("Vedanta",            "VEDL.NS"),
        ("UltraTech Cement",   "ULTRACEMCO.NS"),
        ("Ambuja Cements",     "AMBUJACEM.NS"),
        ("Jindal Steel",       "JINDALSTEL.NS"),
        ("Steel Authority",    "SAIL.NS"),
        ("DLF",                "DLF.NS"),
        ("Prestige Estates",   "PRESTIGE.NS"),
    ],
    "✈️ Travel, Telecom & Media": [
        ("IndiGo (InterGlobe)","INDIGO.NS"),
        ("SpiceJet",           "SPICEJET.NS"),
        ("Bharti Airtel",      "BHARTIARTL.NS"),
        ("Vodafone Idea",      "IDEA.NS"),
        ("Tata Communications","TATACOMM.NS"),
        ("Zee Entertainment",  "ZEEL.NS"),
        ("PVR Inox",           "PVRINOX.NS"),
        ("Indian Hotels",      "INDHOTEL.NS"),
        ("Thomas Cook India",  "THOMASCOOK.NS"),
    ],
}

@st.cache_data(ttl=60)
def _get_market_indices():
    from data_fetcher import fetch_global_indices
    return fetch_global_indices()


# ══════════════════════════════════════════════════
global_indices = _get_market_indices()

# Build small ticker pills for navbar
ticker_html = ""
if global_indices:
    for name, data in list(global_indices.items())[:4]:
        css_c = "gt-up" if data["change"] >= 0 else "gt-dn"
        sign  = "▲" if data["change"] >= 0 else "▼"
        ticker_html += f"""
        <div class="global-ticker">
            <span class="gt-name">{name}</span>
            <span class="{css_c}">{sign} {abs(data['change']):.2f}%</span>
        </div>"""

import streamlit.components.v1 as _components

_navbar_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700;800&display=swap');
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{
        font-family: 'Inter', sans-serif;
        background: rgba(11,15,26,0.96);
        border-bottom: 1px solid rgba(248,250,252,0.07);
    }}
    .navbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 28px;
        height: 60px;
        gap: 16px;
    }}
    .brand {{ display:flex; align-items:center; gap:11px; flex-shrink:0; }}
    .logo {{
        width:34px; height:34px;
        background: linear-gradient(135deg,#f59e0b,#fbbf24);
        border-radius:9px;
        display:flex; align-items:center; justify-content:center;
        font-size:1rem; font-weight:800; color:#0b0f1a;
        box-shadow:0 3px 12px rgba(245,158,11,0.3);
    }}
    .brand-text {{ line-height:1.2; }}
    .brand-name {{ font-size:1.1rem; font-weight:700; color:#f1f5f9; letter-spacing:-0.02em; }}
    .brand-name span {{ color:#f59e0b; }}
    .brand-tag {{ font-size:0.6rem; color:#64748b; font-weight:500; letter-spacing:0.08em; text-transform:uppercase; }}
    .tickers {{ display:flex; align-items:center; gap:6px; flex-wrap:wrap; overflow:hidden; }}
    .ticker-pill {{
        display:flex; align-items:center; gap:5px;
        background:#1a2235; border:1px solid rgba(248,250,252,0.07);
        border-radius:7px; padding:3px 9px; font-size:0.7rem; white-space:nowrap;
    }}
    .t-name {{ color:#64748b; font-weight:500; }}
    .t-up   {{ color:#10b981; font-weight:600; }}
    .t-dn   {{ color:#f43f5e; font-weight:600; }}
    .right  {{ display:flex; align-items:center; gap:10px; flex-shrink:0; }}
    .pill-active {{
        background:rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.3);
        color:#f59e0b; border-radius:7px; padding:5px 12px;
        font-size:0.72rem; font-weight:600; letter-spacing:0.03em;
    }}
    .live {{ display:flex; align-items:center; gap:5px; font-size:0.7rem; color:#10b981; font-weight:500; }}
    .dot {{
        width:7px; height:7px; border-radius:50%; background:#10b981;
        animation: blink 2s infinite;
    }}
    @keyframes blink {{
        0%,100% {{ opacity:1; }} 50% {{ opacity:0.4; }}
    }}
</style>
</head>
<body>
<div class="navbar">
    <div class="brand">
        <div class="logo">S</div>
        <div class="brand-text">
            <div class="brand-name">SWAYIN<span>.AI</span></div>
            <div class="brand-tag">Smart Market Intelligence</div>
        </div>
    </div>
    <div class="tickers">
        {ticker_html}
    </div>
    <div class="right">
        <div class="pill-active">Dashboard</div>
        <div class="live"><div class="dot"></div> Live</div>
    </div>
</div>
</body>
</html>
"""

# Re-build ticker HTML using plain HTML (no Streamlit classes needed inside iframe)
_ticker_inner = ""
if global_indices:
    for _name, _data in list(global_indices.items())[:4]:
        _cls  = "t-up" if _data["change"] >= 0 else "t-dn"
        _sign = "▲" if _data["change"] >= 0 else "▼"
        _ticker_inner += f'<div class="ticker-pill"><span class="t-name">{_name}</span><span class="{_cls}">{_sign} {abs(_data["change"]):.2f}%</span></div>'

_navbar_html = _navbar_html.replace(ticker_html, _ticker_inner)
_components.html(_navbar_html, height=62, scrolling=False)


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR — CONTROLS
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:0 4px 12px 4px;">
        <div style="font-size:1rem; font-weight:700; color:#f1f5f9; letter-spacing:-0.01em;">Configuration</div>
        <div style="font-size:0.72rem; color:#64748b; margin-top:2px;">Adjust parameters and run analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Global Market Context
    with st.expander("🌍 Global Markets", expanded=False):
        if global_indices:
            for name, data in global_indices.items():
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.caption(name)
                with col2:
                    color = "#10b981" if data["change"] >= 0 else "#f43f5e"
                    sign  = "▲" if data["change"] >= 0 else "▼"
                    st.markdown(
                        f"<span style='color:{color}; font-size:0.78rem; font-weight:600; "
                        f"font-family:JetBrains Mono,monospace;'>{sign} {abs(data['change']):.2f}%</span>",
                        unsafe_allow_html=True
                    )
        else:
            st.caption("Global data unavailable.")

    st.markdown('<div class="sidebar-section-title">Market Selection</div>', unsafe_allow_html=True)

    category = st.selectbox(
        "Sector / Category",
        list(INDIAN_MARKET.keys()),
        index=0,
        label_visibility="collapsed",
        help="Choose a market segment",
    )

    options_in_cat = INDIAN_MARKET[category]
    labels  = [f"{label}  ({ticker})" for label, ticker in options_in_cat]
    tickers = [ticker for _, ticker in options_in_cat]

    chosen_idx = st.selectbox(
        "Stock / Index",
        range(len(labels)),
        format_func=lambda i: labels[i],
        index=0,
        label_visibility="collapsed",
        help="Select a stock or index",
    )
    symbol = tickers[chosen_idx]
    st.caption(f"Ticker: `{symbol}`")

    st.markdown('<div class="sidebar-section-title">Data Settings</div>', unsafe_allow_html=True)

    interval = st.radio(
        "Interval",
        options=["1d", "5m"],
        index=0,
        horizontal=True,
        help="'1d' daily | '5m' 5-minute intraday",
    )

    period = st.selectbox(
        "Historical Period",
        ["1y", "2y", "3y", "5y"],
        index=3,
        help="Amount of historical training data",
    )

    st.markdown('<div class="sidebar-section-title">Model</div>', unsafe_allow_html=True)
    
    chart_type = st.radio(
        "Display Mode",
        ["Line Chart", "Candlestick"],
        index=0,
        horizontal=True,
        help="Choose visualization style"
    )

    retrain = st.checkbox("Force Retrain", value=False)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)

    st.divider()
    if st.button("🚪  Secure Logout", use_container_width=True, type="secondary"):
        st.session_state.logged_in = False
        st.session_state.auth_mode = "landing"
        st.rerun()

    st.markdown("""
    <div style="font-size:0.72rem; color:#64748b; line-height:1.7; padding:0 2px; margin-top:10px;">
        <div style="color:#94a3b8; font-weight:600; margin-bottom:6px;">About SWAYIN.AI</div>
        Deep Stacked LSTM (3 layers · 100‑100‑50 units)<br>
        Trained on up to 5 years of market data<br>
        <br>
        <span style="color:#10b981;">●</span> Data via yfinance — no API key needed
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  WELCOME SCREEN
# ══════════════════════════════════════════════════════════════════════
if not run_btn and "results" not in st.session_state:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">Powered by Deep LSTM · Live Market Data</div>
        <div class="hero-title">Smart Predictions for<br><span>Indian Markets</span></div>
        <div class="hero-subtitle">
            Select a stock or index from the sidebar, configure your<br>
            parameters, and let AI do the heavy lifting.
        </div>
        <div class="hero-features">
            <div class="hero-feat">
                <div class="hero-feat-icon">🧠</div>
                <div class="hero-feat-label">Deep LSTM<br>Prediction</div>
            </div>
            <div class="hero-feat">
                <div class="hero-feat-icon">📰</div>
                <div class="hero-feat-label">News Sentiment<br>Analysis</div>
            </div>
            <div class="hero-feat">
                <div class="hero-feat-icon">🔬</div>
                <div class="hero-feat-label">Technical<br>Indicators</div>
            </div>
            <div class="hero-feat">
                <div class="hero-feat-icon">📐</div>
                <div class="hero-feat-label">Explainable<br>AI (XAI)</div>
            </div>
            <div class="hero-feat">
                <div class="hero-feat-icon">📈</div>
                <div class="hero-feat-label">Backtesting<br>Engine</div>
            </div>
        </div>
        <div class="hero-hint">← Use the sidebar to get started</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════
#  RUN PIPELINE
# ══════════════════════════════════════════════════════════════════════
if run_btn or "results" in st.session_state:
    if run_btn:
        _get_model_and_preds.clear()
        with st.spinner("Running full intelligence pipeline…"):
            r = _get_model_and_preds(symbol, interval, period, retrain)
        st.session_state["results"] = r

    r            = st.session_state["results"]
    df           = r["df"]
    train_pred   = r["train_pred"]
    test_pred    = r["test_pred"]
    actual_train = r["actual_train"]
    actual_test  = r["actual_test"]
    history      = r["history"]
    curr         = r["current_price"]
    nd           = r["next_day"]
    nm           = r["next_5min"]
    metrics      = r["metrics"]
    bt           = r["bt_result"]
    train_size   = r["train_size"]
    from config  import TIME_STEP

    signal  = nd["Signal"]
    pred_nd = nd["Predicted Next-Day Close"]
    conf    = nd.get("Confidence (%)", 50)
    chg     = nd.get("Change (%)", 0)

    # ── Page header ─────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-bottom:24px;">
        <div style="font-size:0.7rem; font-weight:600; color:#f59e0b;
                    text-transform:uppercase; letter-spacing:0.1em; margin-bottom:4px;">
            Analysis Report
        </div>
        <div style="font-size:1.55rem; font-weight:800; color:#f1f5f9;
                    letter-spacing:-0.03em; line-height:1.2;">
            {symbol}
            <span style="font-size:0.85rem; font-weight:500; color:#64748b;
                         letter-spacing:0; margin-left:10px;">
                {interval.upper()} · {period}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    #  SECTION 1 — KEY METRICS
    # ══════════════════════════════════════════════════════════════
    st.markdown('<div class="section-label">Performance Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Current Price", f"₹{curr:,.2f}")
    with c2:
        delta = pred_nd - curr
        st.metric("Next-Day Forecast", f"₹{pred_nd:,.2f}", delta=f"{delta:+.2f}")
    with c3:
        st.metric("RMSE", f"{metrics['RMSE']:.2f}")
    with c4:
        st.metric("MAPE", f"{metrics['MAPE (%)']:.2f}%")
    with c5:
        st.metric("Dir. Accuracy", f"{metrics['Directional Accuracy']:.1f}%")

    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    #  SECTION 2 — SIGNAL + CONFIDENCE
    # ══════════════════════════════════════════════════════════════
    st.markdown('<div class="section-label">Trade Signal</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI Recommendation</div>', unsafe_allow_html=True)

    col_sig, col_conf, col_adv = st.columns([1, 2, 2])

    with col_sig:
        badge_class = "signal-buy" if "BUY" in signal else "signal-sell"
        st.markdown(f'<div class="{badge_class}">{signal}</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align:center; margin-top:10px;">
            <span style="font-size:0.72rem; color:#64748b;">Expected Move</span><br>
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.9rem;
                         font-weight:600; color:#f1f5f9;">{chg:+.4f}%</span>
        </div>
        """, unsafe_allow_html=True)

    with col_conf:
        st.markdown(f"""
        <div class="card" style="height:100%;">
            <div style="font-size:0.68rem; color:#64748b; text-transform:uppercase;
                        letter-spacing:0.08em; margin-bottom:6px; font-weight:600;">
                Model Confidence
            </div>
            <div style="font-size:2rem; font-weight:800; font-family:'JetBrains Mono',monospace;
                        color:#f59e0b; letter-spacing:-0.02em;">{conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(min(conf, 100)))

    with col_adv:
        sent        = r.get("sentiment", {"score": 0.0, "label": "NEUTRAL", "headlines": []})
        s_label     = sent.get("label", "NEUTRAL")
        sig_type    = "BULLISH" if "BUY" in signal else "BEARISH"
        atr_val     = df["ATR"].iloc[-1] if "ATR" in df.columns else curr * 0.01

        if sig_type == "BULLISH" and s_label == "BULLISH":
            advice = "Strong Convergence — Momentum and news sentiment align. High probability long setup."
            adv_color = "#10b981"
        elif sig_type == "BEARISH" and s_label == "BEARISH":
            advice = "Bearish Confluence — Technical breakdown confirmed by negative news sentiment."
            adv_color = "#f43f5e"
        elif sig_type != s_label and s_label != "NEUTRAL":
            advice = "Mixed Signals — Sentiment and technicals are diverging. Use tight risk controls."
            adv_color = "#f59e0b"
        else:
            advice = "Neutral Setup — Follow technical levels with a defined stop loss."
            adv_color = "#94a3b8"

        sl_price = curr - (atr_val * 1.5)
        tp_price = curr + (atr_val * 2.0)

        st.markdown(f"""
        <div class="card">
            <div style="font-size:0.68rem; font-weight:600; color:#64748b;
                        text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;">
                🤖 AI Strategy Advisor
            </div>
            <div style="font-size:0.82rem; color:{adv_color}; font-weight:500;
                        line-height:1.5; margin-bottom:14px;">{advice}</div>
            <div class="risk-row">
                <div class="risk-chip">
                    <div class="rc-label">Stop Loss</div>
                    <div class="rc-value rc-sl">₹{sl_price:,.1f}</div>
                </div>
                <div class="risk-chip">
                    <div class="rc-label">Take Profit</div>
                    <div class="rc-value rc-tp">₹{tp_price:,.1f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    #  SECTION 3 — AI INTELLIGENCE MATRIX
    # ══════════════════════════════════════════════════════════════
    st.markdown('<div class="section-label">Intelligence Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI Intelligence Matrix</div>', unsafe_allow_html=True)

    intel_l, intel_m, intel_r = st.columns(3)

    with intel_l:
        s_score     = sent["score"]
        sent_cls    = "s-bullish" if "BULL" in s_label else "s-bearish" if "BEAR" in s_label else "s-neutral"
        st.markdown(f"""
        <div class="sentiment-card">
            <div class="s-label">NLP News Sentiment</div>
            <div class="s-value {sent_cls}">{s_label}</div>
            <div class="s-score">VADER Score: {s_score:+.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("📰 Live Financial News API"):
            if sent.get("headlines"):
                for h in sent["headlines"][:6]:
                    if isinstance(h, dict):
                        # Advanced NLP Rendering
                        title = h.get("title", "")
                        pub = h.get("publisher", "Yahoo Finance API")
                        link = h.get("link", "#")
                        score = h.get("score", 0.0)
                        
                        s_color = "#10b981" if score > 0 else "#f43f5e" if score < 0 else "#94a3b8"
                        
                        st.markdown(f"""
                        <div style='padding:12px 0; border-bottom:1px solid rgba(255,255,255,0.05);'>
                            <a href='{link}' target='_blank' style='text-decoration:none; color:#f1f5f9; font-size:0.85rem; font-weight:600; display:block; margin-bottom:6px; line-height:1.4;'>{title}</a>
                            <div style='display:flex; justify-content:space-between; align-items:center;'>
                                <span style='font-size:0.7rem; color:#64748b; font-weight:600; text-transform:uppercase;'>{pub}</span>
                                <span style='font-size:0.72rem; color:{s_color}; font-family:"JetBrains Mono",monospace; font-weight:700;'>NLP: {score:+.2f}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Fallback for old cache
                        st.markdown(f"<div style='font-size:0.8rem; color:#94a3b8; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.05);'>• {h}</div>", unsafe_allow_html=True)
            else:
                st.caption("No recent news available from the API.")

    with intel_m:
        xai_data = r.get("xai", {})
        if xai_data:
            drivers = list(xai_data.keys())[:5]
            values  = list(xai_data.values())[:5]
            norm_v  = [abs(v) for v in values]
            max_v   = max(norm_v) if max(norm_v) > 0 else 1
            colors  = ["#f59e0b" if v == max(norm_v) else "#38bdf8" for v in norm_v]

            fig_xai = go.Figure(go.Bar(
                x=norm_v, y=drivers, orientation="h",
                marker=dict(color=colors,
                            line=dict(width=0)),
                text=[f"{v:.4f}" for v in norm_v],
                textposition="outside",
                textfont=dict(size=10, color="#94a3b8", family="JetBrains Mono"),
            ))
            fig_xai.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=200,
                margin=dict(l=0, r=60, t=6, b=6),
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(autorange="reversed", tickfont=dict(size=11, family="Inter", color="#94a3b8")),
                showlegend=False,
            )
            st.plotly_chart(fig_xai, use_container_width=True)
            st.caption("Decision Drivers (Explainable AI)")

    with intel_r:
        # Model confidence breakdown as a donut-style progress
        mape_score  = metrics.get("MAPE (%)", 5)
        dir_acc     = metrics.get("Directional Accuracy", 50)

        for label, val, color in [
            ("Directional Accuracy", dir_acc, "#10b981"),
            ("Confidence Score",     conf,    "#f59e0b"),
            ("MAPE (lower=better)",  max(0, 100-mape_score), "#38bdf8"),
        ]:
            bar_width = int(min(val, 100))
            st.markdown(f"""
            <div style="margin-bottom:14px;">
                <div style="display:flex; justify-content:space-between;
                            font-size:0.75rem; margin-bottom:5px;">
                    <span style="color:#94a3b8; font-weight:500;">{label}</span>
                    <span style="color:{color}; font-family:'JetBrains Mono',monospace;
                                 font-weight:600;">{val:.1f}%</span>
                </div>
                <div style="height:6px; background:rgba(255,255,255,0.06);
                            border-radius:99px; overflow:hidden;">
                    <div style="width:{bar_width}%; height:100%;
                                background:{color}; border-radius:99px;
                                transition:width 0.8s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    #  SECTION 4 — PRICE CHART
    # ══════════════════════════════════════════════════════════════
    st.markdown('<div class="section-label">Price History</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Actual vs Predicted Price</div>', unsafe_allow_html=True)

    n_train = train_size - TIME_STEP
    dates   = df.index.tolist()

    train_series = [None] * len(dates)
    test_series  = [None] * len(dates)
    for i, v in enumerate(train_pred):
        idx = TIME_STEP + i
        if idx < len(dates):
            train_series[idx] = float(v)
    start_test = TIME_STEP + n_train
    for i, v in enumerate(test_pred):
        idx = start_test + i
        if idx < len(dates):
            test_series[idx] = float(v)

    fig_main = go.Figure()
    if chart_type == "Candlestick":
        fig_main.add_trace(go.Candlestick(
            x=dates,
            open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="Market Data",
            increasing_line_color="#10b981", decreasing_line_color="#f43f5e"
        ))
    else:
        fig_main.add_trace(go.Scatter(
            x=dates, y=df["Close"].tolist(),
            name="Actual Close",
            line=dict(color="#94a3b8", width=1.5),
        ))

    fig_main.add_trace(go.Scatter(
        x=dates, y=train_series,
        name="Train Prediction",
        line=dict(color="#38bdf8", width=1.8, dash="dot"),
    ))
    fig_main.add_trace(go.Scatter(
        x=dates, y=test_series,
        name="Test Prediction",
        line=dict(color="#f59e0b", width=2.2),
    ))
    if "MA50" in df.columns:
        fig_main.add_trace(go.Scatter(
            x=dates, y=df["MA50"].tolist(),
            name="MA 50", line=dict(color="#e879f9", width=1.2, dash="dot"), opacity=0.7,
        ))
    if "EMA50" in df.columns:
        fig_main.add_trace(go.Scatter(
            x=dates, y=df["EMA50"].tolist(),
            name="EMA 50", line=dict(color="#34d399", width=1.2, dash="dash"), opacity=0.7,
        ))

    # Add current price reference line
    fig_main.add_hline(
        y=curr,
        line=dict(color="#f59e0b", dash="longdash", width=1),
        annotation_text=f"  Now ₹{curr:,.0f}",
        annotation_font=dict(color="#f59e0b", size=11),
    )

    fig_main.update_layout(
        template     = "plotly_dark",
        paper_bgcolor= "#111827",
        plot_bgcolor  = "#0b0f1a",
        height        = 440,
        legend        = dict(orientation="h", yanchor="bottom", y=1.02,
                             bgcolor="rgba(0,0,0,0)", font=dict(size=12, color="#94a3b8")),
        margin        = dict(l=12, r=12, t=12, b=12),
        xaxis         = dict(
            rangeslider=dict(visible=True, bgcolor="#111827", thickness=0.04),
            gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(color="#64748b", size=11),
        ),
        yaxis         = dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#64748b", size=11)),
    )
    st.plotly_chart(fig_main, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    #  SECTION 5 — TECHNICAL INDICATORS
    # ══════════════════════════════════════════════════════════════
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Market Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Technical Indicators</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["  RSI (14)  ", "  MACD  ", "  Bollinger Bands  ", "  Volume  "])

    # Base chart style — NO 'yaxis' or 'margin' here to avoid duplicate-kwarg errors
    _CL = dict(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0b0f1a",
    )
    _GRID_X  = dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#64748b", size=10))
    _GRID_Y  = dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#64748b", size=10))
    _MARGIN  = dict(l=12, r=12, t=16, b=12)

    with tab1:
        if "RSI14" in df.columns:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=dates, y=df["RSI14"].tolist(),
                name="RSI 14",
                line=dict(color="#e879f9", width=1.8),
                fill="tozeroy", fillcolor="rgba(232,121,249,0.04)",
            ))
            fig_rsi.add_hline(y=70, line=dict(color="#f43f5e", dash="dash", width=1),
                              annotation_text="Overbought 70",
                              annotation_font=dict(color="#f43f5e", size=10))
            fig_rsi.add_hline(y=30, line=dict(color="#10b981", dash="dash", width=1),
                              annotation_text="Oversold 30",
                              annotation_font=dict(color="#10b981", size=10))
            fig_rsi.update_layout(
                height=300,
                margin=_MARGIN,
                xaxis={**_GRID_X},
                yaxis={**_GRID_Y, "range": [0, 100]},
                **_CL,
            )
            st.plotly_chart(fig_rsi, use_container_width=True)
        else:
            st.info("RSI data not available for this instrument.")

    with tab2:
        macd_cols = ["MACD", "MACD_Signal", "MACD_Hist"]
        if all(c in df.columns for c in macd_cols):
            fig_macd = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.65, 0.35], vertical_spacing=0.06,
            )
            fig_macd.add_trace(go.Scatter(
                x=dates, y=df["MACD"].tolist(),
                name="MACD", line=dict(color="#38bdf8", width=1.8),
            ), row=1, col=1)
            fig_macd.add_trace(go.Scatter(
                x=dates, y=df["MACD_Signal"].tolist(),
                name="Signal", line=dict(color="#f43f5e", width=1.5),
            ), row=1, col=1)
            hist_colors = ["#10b981" if v >= 0 else "#f43f5e" for v in df["MACD_Hist"].tolist()]
            fig_macd.add_trace(go.Bar(
                x=dates, y=df["MACD_Hist"].tolist(),
                name="Histogram", marker_color=hist_colors, opacity=0.7,
            ), row=2, col=1)
            fig_macd.update_layout(
                height=340,
                margin=_MARGIN,
                xaxis=dict(**_GRID_X),
                xaxis2=dict(**_GRID_X),
                yaxis=dict(**_GRID_Y),
                yaxis2=dict(**_GRID_Y),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11)),
                **_CL,
            )
            st.plotly_chart(fig_macd, use_container_width=True)
        else:
            st.info("MACD data not available for this instrument.")

    with tab3:
        bb_cols = ["BB_Upper", "BB_Middle", "BB_Lower"]
        if all(c in df.columns for c in bb_cols):
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(
                x=dates, y=df["BB_Upper"].tolist(),
                name="Upper Band", line=dict(color="#f43f5e", dash="dot"), opacity=0.8,
            ))
            fig_bb.add_trace(go.Scatter(
                x=dates, y=df["BB_Middle"].tolist(),
                name="Middle (MA20)", line=dict(color="#f59e0b"),
            ))
            fig_bb.add_trace(go.Scatter(
                x=dates, y=df["BB_Lower"].tolist(),
                name="Lower Band", line=dict(color="#10b981", dash="dot"),
                fill="tonexty", fillcolor="rgba(16,185,129,0.04)", opacity=0.8,
            ))
            fig_bb.add_trace(go.Scatter(
                x=dates, y=df["Close"].tolist(),
                name="Close", line=dict(color="#94a3b8", width=1.2),
            ))
            fig_bb.update_layout(
                height=320,
                margin=_MARGIN,
                xaxis={**_GRID_X},
                yaxis={**_GRID_Y},
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11)),
                **_CL,
            )
            st.plotly_chart(fig_bb, use_container_width=True)
        else:
            st.info("Bollinger Band data not available for this instrument.")

    with tab4:
        if "Volume" in df.columns:
            fig_vol = go.Figure()
            colors_vol = ["#10b981" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#f43f5e" 
                          for i in range(len(df))]
            fig_vol.add_trace(go.Bar(
                x=dates, y=df["Volume"].tolist(),
                name="Volume",
                marker_color=colors_vol,
                opacity=0.8
            ))
            fig_vol.update_layout(
                height=300,
                margin=_MARGIN,
                xaxis={**_GRID_X},
                yaxis={**_GRID_Y},
                **_CL,
            )
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # Add Volume Stats
            v_avg = df["Volume"].mean()
            v_curr = df["Volume"].iloc[-1]
            st.markdown(f"""
            <div style="display:flex; gap:20px; margin-top:10px;">
                <div style="font-size:0.8rem; color:#94a3b8;">Average Volume: <span style="color:#f1f5f9; font-weight:600;">{v_avg:,.0f}</span></div>
                <div style="font-size:0.8rem; color:#94a3b8;">Current Volume: <span style="color:#f1f5f9; font-weight:600;">{v_curr:,.0f}</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Volume data not available.")

    # ══════════════════════════════════════════════════════════════
    #  SECTION 6 — 5-MINUTE SIMULATION
    # ══════════════════════════════════════════════════════════════
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Intraday Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">5-Minute Price Simulation</div>', unsafe_allow_html=True)

    sim_prices = nm.get("Simulated 5-Min Prices", {})
    if sim_prices:
        sim_df = pd.DataFrame.from_dict(sim_prices, orient="index", columns=["Predicted Price"])
        sim_df.index.name = "Time Step"
        sim_df["vs Current"] = sim_df["Predicted Price"] - curr
        sim_df["Signal"]     = sim_df["vs Current"].apply(lambda x: "📈 BUY" if x > 0 else "📉 SELL")

        sim_x = list(sim_df.index)
        sim_y = sim_df["Predicted Price"].tolist()

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=sim_x, y=sim_y,
            name="Predicted",
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2.2),
            marker=dict(size=7, color="#f59e0b", line=dict(width=1.5, color="#111827")),
            fill="tozeroy", fillcolor="rgba(245,158,11,0.06)",
        ))
        fig_sim.add_hline(
            y=curr,
            line=dict(color="#94a3b8", dash="longdash", width=1),
            annotation_text=f"  Current ₹{curr:,.0f}",
            annotation_font=dict(color="#94a3b8", size=10),
        )
        fig_sim.update_layout(
            height=300,
            margin=_MARGIN,
            xaxis={**_GRID_X},
            yaxis={**_GRID_Y},
            **_CL,
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        st.dataframe(
            sim_df.style
                  .format({"Predicted Price": "₹{:,.2f}", "vs Current": "{:+.2f}"})
                  .applymap(lambda v: "color:#10b981" if "BUY" in str(v) else "color:#f43f5e",
                            subset=["Signal"]),
            use_container_width=True,
        )

    # ══════════════════════════════════════════════════════════════
    #  SECTION 7 — BACKTESTING
    # ══════════════════════════════════════════════════════════════
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Strategy Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Backtesting Results</div>', unsafe_allow_html=True)

    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1:
        st.metric("Initial Capital", f"₹{bt['Initial Capital']:,.0f}")
    with bc2:
        delta_cap = bt["Final Capital"] - bt["Initial Capital"]
        st.metric("Final Capital",  f"₹{bt['Final Capital']:,.0f}", delta=f"{delta_cap:+,.0f}")
    with bc3:
        st.metric("Total Return",   f"{bt['Total Return (%)']:.2f}%")
    with bc4:
        st.metric("Win Rate",       f"{bt['Win Rate (%)']:.1f}%")

    if bt.get("Portfolio Values"):
        fig_bt = go.Figure()
        pv     = bt["Portfolio Values"]
        colors_bt = ["#10b981" if v >= bt["Initial Capital"] else "#f43f5e" for v in pv]
        fig_bt.add_trace(go.Scatter(
            y=pv, name="Portfolio Value",
            line=dict(color="#10b981", width=2),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.06)",
        ))
        fig_bt.add_hline(y=bt["Initial Capital"],
                         line=dict(color="#64748b", dash="dash", width=1),
                         annotation_text="  Initial Capital",
                         annotation_font=dict(color="#64748b", size=10))
        fig_bt.update_layout(
            height=300,
            margin=_MARGIN,
            xaxis={**_GRID_X},
            yaxis={**_GRID_Y},
            **_CL,
        )
        st.plotly_chart(fig_bt, use_container_width=True)

    # Export Trade Log CSV
    if bt.get("Trade Log"):
        trade_df = pd.DataFrame(bt["Trade Log"])
        csv_trades = trade_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📊 Download Backtest Trade Log",
            data=csv_trades,
            file_name=f"SWAYIN_Backtest_{symbol}_{interval}.csv",
            mime="text/csv",
        )

    # ══════════════════════════════════════════════════════════════
    #  SECTION 8 — MODEL EVALUATION TABLE
    # ══════════════════════════════════════════════════════════════
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Evaluation Metrics</div>', unsafe_allow_html=True)

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
    metrics_df.index.name = "Metric"
    st.dataframe(
        metrics_df.style.format({"Value": "{:.4f}"}),
        use_container_width=True,
    )
    
    # Export Metrics CSV
    csv_metrics = metrics_df.to_csv().encode("utf-8")
    st.download_button(
        label="📥 Download Metrics Report",
        data=csv_metrics,
        file_name=f"SWAYIN_Metrics_{symbol}_{interval}.csv",
        mime="text/csv",
    )

    # ══════════════════════════════════════════════════════════════
    #  SECTION 9 — TRAINING HISTORY
    # ══════════════════════════════════════════════════════════════
    if history is not None:
        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Training</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Model Training History</div>', unsafe_allow_html=True)
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=history.history["loss"],
            name="Train Loss", line=dict(color="#f59e0b", width=1.8),
        ))
        fig_loss.add_trace(go.Scatter(
            y=history.history["val_loss"],
            name="Val Loss", line=dict(color="#10b981", width=1.8, dash="dot"),
        ))
        fig_loss.update_layout(
            height=280,
            yaxis_type="log",
            xaxis_title="Epoch",
            yaxis_title="Loss (log scale)",
            margin=_MARGIN,
            xaxis={**_GRID_X},
            yaxis={**_GRID_Y},
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11)),
            **_CL,
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    # ── Footer ───────────────────────────────────────────────────
    st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding:24px 0;
                border-top:1px solid rgba(248,250,252,0.07);">
        <div style="font-size:0.72rem; color:#334155; font-weight:500;">
            SWAYIN.AI &nbsp;·&nbsp; Deep LSTM Intelligence &nbsp;·&nbsp;
            TensorFlow &amp; Streamlit &nbsp;·&nbsp; Data via yfinance
        </div>
    </div>
    """, unsafe_allow_html=True)
