import os
import sys
import random
import math
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from werkzeug.utils import secure_filename
import joblib

# --- 1. DİNAMİK YAPI VE YOLLAR ---
# Uygulamanın çalıştığı ana dizini tespit et
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Klasör yollarını ana dizine göre belirle
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DB_DIR = os.path.join(BASE_DIR, 'database')
DB_PATH = os.path.join(DB_DIR, 'arizalar.db')
MODEL_PATH = os.path.join(BASE_DIR, "text_model.pkl")

# Eksik klasörleri otomatik oluştur (Hata engelleyici)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# --- 2. FLASK YAPILANDIRMASI ---
app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))

app.secret_key = 'AATS_TARSUS_OZEL_GIZLI_KEY_2026'  # Güvenlik için eklendi
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

db = SQLAlchemy(app)
logging.basicConfig(level=logging.INFO)

# --- 3. MODEL YÜKLEME ---
text_model = None
if os.path.exists(MODEL_PATH):
    try:
        text_model = joblib.load(MODEL_PATH)
    except Exception as e:
        logging.error(f"Model yükleme hatası: {e}")


# --- 4. VERİTABANI MODELİ ---
class Arizalar(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    aciklama = db.Column(db.String(255))
    mahalle = db.Column(db.String(100))
    sokak = db.Column(db.String(100))
    tur = db.Column(db.String(50))
    tahmini_sure = db.Column(db.Float)
    oncelik = db.Column(db.String(50))
    foto = db.Column(db.String(255))
    konum_lat = db.Column(db.Float)
    konum_lng = db.Column(db.Float)
    durum = db.Column(db.String(50), default="Beklemede")
    sikayet_sayisi = db.Column(db.Integer, default=1)
    tarih = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()


# --- 5. YARDIMCI FONKSİYONLAR (NLP & GEOMETRİ) ---
def tr_lower(text):
    if not text: return ""
    return text.replace('I', 'ı').replace('İ', 'i').lower()


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def determine_priority(text):
    t = tr_lower(text)
    if any(x in t for x in ["tehlike", "acil", "yaralı", "kaza", "elektrik", "patlama", "yangın", "açık", "çukur"]):
        return "Yüksek"
    if any(x in t for x in ["taşma", "koku", "patlak", "kesinti", "akıyor", "tıkan", "kırık"]):
        return "Orta"
    return "Düşük"


def predict_text_type(text):
    if not text: return "Genel"
    t = tr_lower(text)
    if any(k in t for k in ["kanal", "logar", "lağım", "rögar", "pis su", "gider", "tıkalı", "foseptik"]):
        return "Kanalizasyon"

    if text_model:
        try:
            pred = text_model.predict([text])[0]
            if pred != "Genel":
                return "Elektrik" if pred == "Aydınlatma" else pred
        except:
            pass

    if any(x in t for x in ["lamba", "ışık", "elektrik", "direk", "trafo"]): return "Elektrik"
    if any(x in t for x in ["boru", "patlak", "su akıyor", "vana", "musluk"]): return "Su"
    if any(x in t for x in ["yol", "çukur", "asfalt", "kaldırım"]): return "Yol"
    if any(x in t for x in ["çöp", "konteyner", "pislik", "atık"]): return "Çöp"
    return "Genel"


# --- 6. FUZZY LOGIC (AKILLI SÜRE) ---
def calculate_eta_smart(tur, mahalle, oncelik):
    base_hours = {"Su": 4.0, "Kanalizasyon": 5.0, "Elektrik": 2.5, "Yol": 24.0, "Çöp": 1.5, "Park/Bahçe": 6.0,
                  "Trafik": 2.0, "Genel": 3.0}
    base_t = base_hours.get(tur, 3.0)

    mahalle_yuku = Arizalar.query.filter_by(tur=tur, mahalle=mahalle).filter(Arizalar.durum != "Tamamlandı").count()
    genel_yuk = Arizalar.query.filter_by(tur=tur).filter(Arizalar.durum != "Tamamlandı").count()

    is_yuku_skoru = min((mahalle_yuku * 0.7 + genel_yuk * 0.3) / 10.0, 1.2)
    oncelik_indirimi = {"Düşük": 0.0, "Orta": 0.3, "Yüksek": 0.6}.get(oncelik, 0.0)

    sinerji_indirimi = 0
    if mahalle_yuku > 1:
        sinerji_indirimi = min((mahalle_yuku - 1) * 0.15, 0.45)

    multiplier = 1.0 + is_yuku_skoru - oncelik_indirimi - sinerji_indirimi
    return round(base_t * max(0.5, multiplier) * 2) / 2


# --- 7. ROTALAR ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    # Personel kontrolü (Daha önce konuştuğumuz güvenlik)
    if not session.get('is_admin'):
        return redirect(url_for('login'))
    return render_template('dashboard.html')


@app.route('/personel-giris', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form.get('sifre') == "TarsusAATS2026":
            session['is_admin'] = True
            return redirect(url_for('dashboard'))
        return "Hatalı şifre!"
    return '''<div style="text-align:center; padding:50px; font-family:sans-serif;">
                <h3>AATS Personel Girişi</h3>
                <form method="post"> Şifre: <input type="password" name="sifre"> <input type="submit" value="Giriş"></form>
              </div>'''


@app.route('/logout')
def logout():
    session.pop('is_admin', None)
    return redirect(url_for('index'))


@app.route('/submit', methods=['POST'])
def submit():
    try:
        aciklama = request.form.get('aciklama', "")
        mahalle = request.form.get('mahalle', "")
        sokak = request.form.get('sokak', "")
        lat, lng = float(request.form.get('lat', 36.9177)), float(request.form.get('lng', 34.8928))

        tur = predict_text_type(aciklama)
        oncelik = determine_priority(aciklama)

        # Merge Logic
        active = Arizalar.query.filter_by(tur=tur).filter(Arizalar.durum != "Tamamlandı").all()
        for ex in active:
            if haversine_distance(lat, lng, ex.konum_lat, ex.konum_lng) < 50:
                ex.sikayet_sayisi += 1
                ex.oncelik = "Yüksek" if ex.sikayet_sayisi >= 3 else ("Orta" if ex.sikayet_sayisi >= 2 else ex.oncelik)
                ex.tahmini_sure = calculate_eta_smart(tur, mahalle, ex.oncelik)
                db.session.commit()
                return jsonify({"durum": "merged", "yeni_sayi": ex.sikayet_sayisi})

        foto_path = None
        foto = request.files.get('foto')
        if foto and foto.filename:
            fname = secure_filename(f"{datetime.now().timestamp()}_{foto.filename}")
            foto.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))
            foto_path = f"/uploads/{fname}"

        tahmini_sure = calculate_eta_smart(tur, mahalle, oncelik)
        yeni = Arizalar(aciklama=aciklama, mahalle=mahalle, sokak=sokak, tur=tur,
                        tahmini_sure=tahmini_sure, oncelik=oncelik, foto=foto_path, konum_lat=lat, konum_lng=lng)
        db.session.add(yeni)
        db.session.commit()
        return jsonify({"durum": "ok", "tur": tur, "oncelik": oncelik, "tahmini_sure": yeni.tahmini_sure})
    except Exception as e:
        return jsonify({"durum": "error", "message": str(e)}), 500


@app.route('/data')
def get_data():
    data = []
    for a in Arizalar.query.order_by(Arizalar.sikayet_sayisi.desc()).all():
        data.append({
            "id": a.id, "tur": a.tur, "aciklama": a.aciklama, "mahalle": a.mahalle,
            "sokak": a.sokak, "oncelik": a.oncelik, "tahmini_sure": a.tahmini_sure,
            "durum": a.durum, "konum_lat": a.konum_lat, "konum_lng": a.konum_lng,
            "foto": a.foto, "sikayet_sayisi": a.sikayet_sayisi,
            "tarih": a.tarih.isoformat() + 'Z' if a.tarih else None
        })
    return jsonify(data)


@app.route('/update_durum/<int:id>', methods=['POST'])
def update_durum(id):
    ariza = Arizalar.query.get_or_404(id)
    if ariza.durum != "Tamamlandı":
        ariza.durum = "Tamamlandı"
        db.session.commit()
        komsular = Arizalar.query.filter_by(mahalle=ariza.mahalle, tur=ariza.tur).filter(
            Arizalar.durum != "Tamamlandı").all()
        for k in komsular:
            k.tahmini_sure = calculate_eta_smart(k.tur, k.mahalle, k.oncelik)
        db.session.commit()
    return jsonify({"success": True})


@app.route('/stats')
def get_stats():
    toplam = Arizalar.query.count()
    cozulen = Arizalar.query.filter_by(durum="Tamamlandı").count()
    tm = db.session.query(Arizalar.mahalle, func.count(Arizalar.id)).group_by(Arizalar.mahalle).limit(5).all()
    tt = db.session.query(Arizalar.tur, func.count(Arizalar.id)).group_by(Arizalar.tur).all()
    return jsonify({
        "mahalleler": [x[0] for x in tm], "mahalle_sayilari": [x[1] for x in tm],
        "turler": [x[0] for x in tt], "tur_sayilari": [x[1] for x in tt],
        "ozet": {"toplam": toplam, "cozulen": cozulen, "bekleyen": toplam - cozulen}
    })


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# --- 8. SUNUCU BASLATMA ---
if __name__ == "__main__":
    # Render'ın verdiği portu al, bulamazsan 5000 kullan
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
