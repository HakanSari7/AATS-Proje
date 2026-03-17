import sys
import os
import webbrowser
import socket
from threading import Timer
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from datetime import datetime
from werkzeug.utils import secure_filename
import joblib
import logging
import random
import math

# --- AYARLAR ---
if getattr(sys, 'frozen', False):
    INTERNAL_DIR = sys._MEIPASS
    EXTERNAL_DIR = os.path.dirname(sys.executable)
else:
    INTERNAL_DIR = os.path.abspath(os.path.dirname(__file__))
    EXTERNAL_DIR = INTERNAL_DIR

template_dir = os.path.join(INTERNAL_DIR, 'templates')
static_dir = os.path.join(INTERNAL_DIR, 'static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# --- KONFİGÜRASYON ---
UPLOAD_FOLDER = os.path.join(EXTERNAL_DIR, 'uploads')
DB_PATH = os.path.join(EXTERNAL_DIR, 'database', 'arizalar.db')
MODEL_PATH = os.path.join(INTERNAL_DIR, "text_model.pkl")

# Pürüz Giderme: Mobil fotoğraflar için dosya boyutu sınırını 16MB yapalım
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
logging.basicConfig(level=logging.INFO)

# --- MODEL YÜKLEME ---
text_model = None
try:
    if os.path.exists(MODEL_PATH):
        text_model = joblib.load(MODEL_PATH)
except:
    pass


# --- VERİTABANI ---
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


# --- YARDIMCI FONKSİYONLAR ---
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


# --- ÖNCELİK BELİRLEME ---
def determine_priority(text):
    t = tr_lower(text)
    if any(x in t for x in ["tehlike", "acil", "yaralı", "kaza", "elektrik", "patlama", "yangın", "açık", "çukur"]):
        return "Yüksek"
    if any(x in t for x in ["taşma", "koku", "patlak", "kesinti", "akıyor", "tıkan", "kırık"]):
        return "Orta"
    return "Düşük"


# --- TÜR TAHMİNİ ---
def predict_text_type(text):
    if not text: return "Genel"
    t = tr_lower(text)
    if any(k in t for k in
           ["kanal", "logar", "lağım", "lagim", "rögar", "rogar", "pis su", "gider", "tıkalı", "foseptik", "koku"]):
        return "Kanalizasyon"
    if text_model:
        try:
            pred = text_model.predict([text])[0]
            if pred != "Genel":
                if pred == "Aydınlatma": return "Elektrik"
                if pred == "Su" and ("kanal" in t or "pis" in t or "logar" in t): return "Kanalizasyon"
                return pred
        except:
            pass
    elek_keywords = ["lamba", "ışık", "isik", "karanlık", "karanlik", "elektrik", "direk", "sönük", "sonuk", "kablo",
                     "trafo", "sigorta", "voltaj", "kıvılcım", "kivilcim", "ceryan", "şartel", "sarter", "pano",
                     "ampul", "yanmıyor"]
    if any(x in t for x in elek_keywords): return "Elektrik"
    su_keywords = ["boru", "patlak", "sızıyor", "siziyor", "vana", "musluk", "şebeke", "sebeke", "temiz su",
                   "su akıyor", "su kesik", "su yok", "birikiyor", "su gitmiyor"]
    if any(x in t for x in su_keywords): return "Su"
    if any(x in t for x in
           ["yol", "çukur", "cukur", "asfalt", "asvalt", "kaldırım", "kaldirim", "tümsek", "kasis", "yamuk", "bozuk",
            "parke"]): return "Yol"
    cop_keywords = ["çöp", "cop", "konteyner", "kutu", "pislik", "pis ", "temizlik", "temizle", "süpür", "atık",
                    "poşet", "moloz"]
    if any(x in t for x in cop_keywords): return "Çöp"
    if any(
        x in t for x in ["park", "ağaç", "agac", "dal", "peyzaj", "çim", "cim", "ot ", "sulama"]): return "Park/Bahçe"
    if any(x in t for x in ["trafik", "levha", "sinyal", "tabela", "durak", "işaret"]): return "Trafik"
    return "Genel"


# --- REVİZE EDİLEN BULANIK MANTIK (FUZZY LOGIC) ---
def calculate_eta_smart(tur, mahalle, oncelik):
    base_hours = {
        "Su": 4.0, "Kanalizasyon": 5.0, "Elektrik": 2.5,
        "Yol": 24.0, "Çöp": 1.5, "Park/Bahçe": 6.0,
        "Trafik": 2.0, "Genel": 3.0
    }
    base_t = base_hours.get(tur, 3.0)

    # Aktif iş yükü (Sadece 'Tamamlanmadı' olanlar)
    mahalle_yuku = Arizalar.query.filter_by(tur=tur, mahalle=mahalle).filter(Arizalar.durum != "Tamamlandı").count()
    genel_yuk = Arizalar.query.filter_by(tur=tur).filter(Arizalar.durum != "Tamamlandı").count()

    # İŞ YÜKÜ ETKİSİ: Mahallede iş yükü azaldıkça çarpan küçülür (Süre düşer)
    is_yuku_skoru = (mahalle_yuku * 0.7 + genel_yuk * 0.3) / 10.0
    is_yuku_skoru = min(is_yuku_skoru, 1.2)  # En fazla %120 ekleyebilir

    # ÖNCELİK ETKİSİ: Yüksek öncelik her zaman süreyi aşağı çeker
    oncelik_skorlari = {"Düşük": 0.0, "Orta": 0.3, "Yüksek": 0.6}
    oncelik_indirimi = oncelik_skorlari.get(oncelik, 0.0)

    # MAHALLE SİNERJİSİ: Ekipler mahalledeyse diğer işler hızlanır (SENİN İSTEDİĞİN MANTIK)
    sinerji_indirimi = 0
    if mahalle_yuku > 1:
        # Mahallede bekleyen her ek iş için %15 (0.15) 'yol tasarrufu' indirimi
        sinerji_indirimi = min((mahalle_yuku - 1) * 0.15, 0.45)

        # Final çarpan hesabı: Baz Süre + İş Yükü - Öncelik Avantajı - Mahalle Sinerjisi
    multiplier = 1.0 + is_yuku_skoru - oncelik_indirimi - sinerji_indirimi

    # Sürenin 0 veya eksi olmasını engelle (Min 30 dk)
    final_eta = base_t * max(0.5, multiplier)

    return round(final_eta * 2) / 2


# --- ROTALAR ---
@app.route('/')
def index(): return render_template('index.html')


@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html')


@app.route('/manifest.json')
def serve_manifest():
    return send_from_directory(INTERNAL_DIR, 'manifest.json')


@app.route('/sw.js')
def serve_sw():
    return send_from_directory(INTERNAL_DIR, 'sw.js')


@app.route('/submit', methods=['POST'])
def submit():
    try:
        aciklama = request.form.get('aciklama', "")
        mahalle = request.form.get('mahalle', "")
        sokak = request.form.get('sokak', "")
        try:
            lat, lng = float(request.form.get('lat')), float(request.form.get('lng'))
        except:
            lat, lng = 36.9177, 34.8928
        foto = request.files.get('foto')

        tur = predict_text_type(aciklama)
        oncelik = determine_priority(aciklama)

        # Merge Logic
        active = Arizalar.query.filter_by(tur=tur).filter(Arizalar.durum != "Tamamlandı").all()
        for ex in active:
            if haversine_distance(lat, lng, ex.konum_lat, ex.konum_lng) < 50:
                ex.sikayet_sayisi += 1
                if ex.sikayet_sayisi >= 3:
                    ex.oncelik = "Yüksek"
                elif ex.sikayet_sayisi >= 2 and ex.oncelik == "Düşük":
                    ex.oncelik = "Orta"

                ex.tahmini_sure = calculate_eta_smart(tur, mahalle, ex.oncelik)
                db.session.commit()
                return jsonify({"durum": "merged", "yeni_sayi": ex.sikayet_sayisi})

        # Jitter Logic
        offset = 0.00015
        lat += random.uniform(-offset, offset)
        lng += random.uniform(-offset, offset)

        foto_path = None
        if foto and foto.filename:
            fname = secure_filename(f"{datetime.now().timestamp()}_{foto.filename}")
            foto.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))
            foto_path = f"/uploads/{fname}"

        tahmini_sure = calculate_eta_smart(tur, mahalle, oncelik)

        yeni = Arizalar(aciklama=aciklama, mahalle=mahalle, sokak=sokak, tur=tur,
                        tahmini_sure=tahmini_sure,
                        oncelik=oncelik, foto=foto_path, konum_lat=lat, konum_lng=lng)
        db.session.add(yeni)
        db.session.commit()
        return jsonify({"durum": "ok", "tur": tur, "oncelik": oncelik, "tahmini_sure": yeni.tahmini_sure})
    except Exception as e:
        return jsonify({"durum": "error", "message": str(e)}), 500


@app.route('/data')
def get_data():
    data = []
    for a in Arizalar.query.order_by(Arizalar.sikayet_sayisi.desc()).all():
        tarih_str = a.tarih.isoformat() + 'Z' if a.tarih else None
        data.append({
            "id": a.id, "tur": a.tur, "aciklama": a.aciklama, "mahalle": a.mahalle,
            "sokak": a.sokak, "oncelik": a.oncelik, "tahmini_sure": a.tahmini_sure,
            "durum": a.durum, "konum_lat": a.konum_lat, "konum_lng": a.konum_lng,
            "foto": a.foto, "sikayet_sayisi": a.sikayet_sayisi, "tarih": tarih_str
        })
    return jsonify(data)


@app.route('/update_durum/<int:id>', methods=['POST'])
def update_durum(id):
    try:
        ariza = Arizalar.query.get_or_404(id)
        if ariza.durum != "Tamamlandı":
            ariza.durum = "Tamamlandı"
            db.session.commit()  # Önce bu işi bitir ki mahalle yükü düşsün

            # Mahalledeki diğer aynı tür işleri bul ve sürelerini güncelle (Hızlanacaklar!)
            komsular = Arizalar.query.filter_by(mahalle=ariza.mahalle, tur=ariza.tur).filter(
                Arizalar.durum != "Tamamlandı").all()

            for k in komsular:
                k.tahmini_sure = calculate_eta_smart(k.tur, k.mahalle, k.oncelik)

            db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/delete_ariza/<int:id>', methods=['POST'])
def delete_ariza(id):
    try:
        ariza = Arizalar.query.get_or_404(id)
        m_temp, t_temp, d_temp = ariza.mahalle, ariza.tur, ariza.durum
        db.session.delete(ariza)
        db.session.commit()

        if d_temp != "Tamamlandı":
            komsular = Arizalar.query.filter_by(mahalle=m_temp, tur=t_temp).filter(
                Arizalar.durum != "Tamamlandı").all()
            for k in komsular:
                k.tahmini_sure = calculate_eta_smart(k.tur, k.mahalle, k.oncelik)
            db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/stats')
def get_stats():
    try:
        toplam = Arizalar.query.count()
        cozulen = Arizalar.query.filter_by(durum="Tamamlandı").count()
        tm = db.session.query(Arizalar.mahalle, func.count(Arizalar.id)).group_by(Arizalar.mahalle).limit(5).all()
        tt = db.session.query(Arizalar.tur, func.count(Arizalar.id)).group_by(Arizalar.tur).all()
        return jsonify({
            "mahalleler": [x[0] for x in tm], "mahalle_sayilari": [x[1] for x in tm],
            "turler": [x[0] for x in tt], "tur_sayilari": [x[1] for x in tt],
            "ozet": {"toplam": toplam, "cozulen": cozulen, "bekleyen": toplam - cozulen}
        })
    except:
        return jsonify({"mahalleler": [], "turler": [], "ozet": {"toplam": 0, "cozulen": 0, "bekleyen": 0}})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


if __name__ == '__main__':
    PORT = 5000
    if is_port_in_use(PORT):
        webbrowser.open_new(f'http://127.0.0.1:{PORT}/')
    else:
        # Sunumda telefondan bağlanabilmen için host='0.0.0.0' ekledim
        Timer(1.5, lambda: webbrowser.open_new(f'http://127.0.0.1:{PORT}/')).start()
        app.run(host='0.0.0.0', debug=False, port=PORT)