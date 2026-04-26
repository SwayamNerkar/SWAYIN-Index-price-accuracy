import json
import os
import requests

LOCAL_DB_FILE = "users_db.json"
FIREBASE_CONFIG_FILE = "firebase_config.json"

def init_config():
    """Create template config files if they don't exist."""
    if not os.path.exists(FIREBASE_CONFIG_FILE):
        with open(FIREBASE_CONFIG_FILE, "w") as f:
            json.dump({"apiKey": "YOUR_FIREBASE_WEB_API_KEY_HERE"}, f, indent=4)
            
    if not os.path.exists(LOCAL_DB_FILE):
        with open(LOCAL_DB_FILE, "w") as f:
            json.dump({}, f)

def get_firebase_key():
    init_config()
    try:
        with open(FIREBASE_CONFIG_FILE, "r") as f:
            config = json.load(f)
            return config.get("apiKey", "")
    except Exception:
        return ""

def _load_local_db():
    init_config()
    try:
        with open(LOCAL_DB_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_local_db(db):
    try:
        with open(LOCAL_DB_FILE, "w") as f:
            json.dump(db, f, indent=4)
    except Exception as e:
        print(f"Error saving local DB: {e}")

def sign_up(email, password, name=""):
    """
    Signs up a user. Tries Firebase first if configured,
    otherwise falls back to local JSON file database.
    """
    if not email or not password:
        return False, "Missing email or password."

    api_key = get_firebase_key()
    
    if api_key and api_key != "YOUR_FIREBASE_WEB_API_KEY_HERE":
        # Firebase Authentication
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={api_key}"
        payload = {"email": email, "password": password, "returnSecureToken": True}
        r = requests.post(url, json=payload)
        data = r.json()
        if "error" in data:
            return False, data["error"].get("message", "Firebase Signup Failed").replace("_", " ")
        return True, "Account created strictly via Firebase."
    else:
        # Local JSON Database Authentication
        db = _load_local_db()
        if email in db:
            return False, "EMAIL ALREADY EXISTS IN SYSTEM."
        
        db[email] = {
            "password": password, 
            "name": name
        }
        _save_local_db(db)
        return True, "Local database account created successfully."

def sign_in(email, password):
    """
    Logs in a user via Firebase or Local JSON.
    """
    if not email or not password:
        return False, "Missing email or password."

    api_key = get_firebase_key()
    
    if api_key and api_key != "YOUR_FIREBASE_WEB_API_KEY_HERE":
        # Firebase Authentication
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
        payload = {"email": email, "password": password, "returnSecureToken": True}
        r = requests.post(url, json=payload)
        data = r.json()
        if "error" in data:
            return False, data["error"].get("message", "Firebase Login Failed").replace("_", " ")
        return True, "Login Successful (Firebase)."
    else:
        # Local JSON Database Authentication
        db = _load_local_db()
        if email not in db:
            return False, "EMAIL NOT FOUND IN LOCAL DB."
        
        if db[email]["password"] != password:
            return False, "INVALID PASSWORD."
            
        return True, "Login Successful (Local)."
