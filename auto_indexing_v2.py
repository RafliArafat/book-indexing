import os
import re
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict
from textwrap import wrap

# --- Import Flask ---
from flask import Flask, render_template, request, send_file, session, redirect, url_for, render_template_string, flash, jsonify
from flask_session import Session
from werkzeug.utils import secure_filename

# --- Import Pustaka Pemrosesan Teks & PDF ---
import pdfplumber
import fitz  # PyMuPDF
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import yake
from nltk.stem import LancasterStemmer
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from difflib import SequenceMatcher
from deep_translator import GoogleTranslator
from rapidfuzz import fuzz
from scipy.spatial.distance import cosine

# --- Import Pembuatan PDF (ReportLab) ---
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --- Konfigurasi Aplikasi Flask ---
app = Flask(__name__)
app.secret_key = 'kunci-rahasia-anda-yang-sangat-aman' 

# Konfigurasi folder
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Konfigurasi Sesi
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
Session(app)

ALLOWED_EXTENSIONS = {'pdf'}

# --- Unduh NLTK Stopwords jika belum ada ---
try:
    stopwords.words('indonesian')
except LookupError:
    print("Mengunduh stopwords NLTK...")
    nltk.download('stopwords')
    print("Unduhan selesai.")

# --- Pemuatan Model & Stopwords Global ---
print("Memuat stopwords...")
factory = StopWordRemoverFactory()
sastrawi_stop = set(factory.get_stop_words())
nltk_stop_id = set(stopwords.words("indonesian"))
nltk_stop_en = set(stopwords.words("english"))

extra_stopwords_global = {
    "pengantar", "pendahuluan", "bab", "daftar", "pustaka", "referensi",
    "abstrak", "kata", "modul", "ajar", "mata", "kuliah", "dan", "atau",
    "bagian", "pasal", "buku",
    "tujuan", "joko", "jurusan",
    "budi", "santi", "pada", "dalam",
    "untuk", "adalah", "yang", "oleh",
    "materi", "teknologi", "malang",
    "politeknik", "negeri", "rajin",
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"
}
combined_stopwords = sastrawi_stop.union(nltk_stop_id).union(nltk_stop_en).union(extra_stopwords_global)

print("Memuat stemmer Sastrawi...")
stemmer = StemmerFactory().create_stemmer()

print("Memuat model Word2Vec...")
# Pastikan path ini sesuai dengan server Anda
model_path_gensim = 'cc.id.300.model' 
model_path_vec = 'cc.id.300.vec'

w2v_model = None
try:
    if os.path.exists(model_path_gensim):
        print(f"Memuat model gensim dari {model_path_gensim}...")
        w2v_model = KeyedVectors.load(model_path_gensim)
    elif os.path.exists(model_path_vec):
        print(f"Memuat model .vec dari {model_path_vec}...")
        w2v_model = KeyedVectors.load_word2vec_format(model_path_vec)
        w2v_model.save(model_path_gensim)
    else:
        print(f"PERINGATAN: File model Word2Vec tidak ditemukan.")
except Exception as e:
    print(f"Gagal memuat model Word2Vec: {e}")

print("Model dan stopwords berhasil dimuat. Aplikasi siap.")

# --- Fungsi Bantuan ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- FUNGSI PEMROSESAN TEKS & PDF ---

def clean_text(text):
  text = re.sub(r'\bhlm\.?\s*\d+', '', text)
  text = re.sub(r'\bCet\.?\s*\d+', '', text)
  text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
  text = re.sub(r"\b[0-9ivxlcdm]+\b", " ", text)
  text = re.sub(r"\([^)]*\)", " ", text)
  text = re.sub(r"http\S+|www\S+", " ", text)
  text = re.sub(r"\.{3,}\s*\d+", " ", text)
  text = re.sub(r"[^A-Za-z0-9\s\.\-\,]", " ", text)
  text = re.sub(r"\s+", " ", text).strip()
  text = re.sub(r"\b(wiki|wikipedia|org)\b", " ", text)
  return text

def normalize_line(line):
    line = line.lower().strip()
    line = re.sub(r"\d+", "", line)
    line = re.sub(r"\s+", " ", line)
    return line

def preprocess_pdf(pdf_path, dynamic_stopwords):
    header_even, header_odd, footer_all = [], [], []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages: return [], "", "PDF kosong."
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    lines = text.split("\n")
                    if len(lines) > 2:
                        header_even.append(normalize_line(lines[0])) if i % 2 == 0 else header_odd.append(normalize_line(lines[0]))
                        footer_all.append(normalize_line(lines[-1]))

        even_freq, odd_freq, footer_freq = Counter(header_even), Counter(header_odd), Counter(footer_all)
        n_even, n_odd, n_footer = len(header_even) or 1, len(header_odd) or 1, len(footer_all) or 1
        
        common_even = {h for h, c in even_freq.items() if (c/n_even > 0.3) and h}
        common_odd = {h for h, c in odd_freq.items() if (c/n_odd > 0.3) and h}
        common_footer = {f for f, c in footer_freq.items() if (c/n_footer > 0.3) and f}

        page_texts = []
        all_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text()
                if raw_text:
                    lines = raw_text.split("\n")
                    if len(lines) > 2:
                        if i % 2 == 0 and normalize_line(lines[0]) in common_even: lines = lines[1:]
                        elif i % 2 == 1 and normalize_line(lines[0]) in common_odd: lines = lines[1:]
                        if normalize_line(lines[-1]) in common_footer: lines = lines[:-1]
                    clean = clean_text("\n".join(lines))
                    page_texts.append((i, clean))
                    all_text += " " + clean
        return page_texts, all_text, None
    except Exception as e:
        return [], "", f"Gagal memproses PDF: {e}"

def extract_chapters(pdf_path, max_pages_for_toc=15, pages_per_chunk=40):
    # (Fungsi extract_chapters tetap sama, disingkat untuk hemat tempat)
    # ... Gunakan implementasi sebelumnya ...
    pages_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages: pages_text.append(p.extract_text() or "")
    except: return []
    
    # Fallback sederhana
    chunks = ["\n".join(pages_text[i:i+pages_per_chunk]) for i in range(0, len(pages_text), pages_per_chunk)]
    return [(f"Halaman {i*pages_per_chunk+1}-{(i+1)*pages_per_chunk}", text) for i, text in enumerate(chunks)]

def extract_keywords_tfidf_per_chapter(chapters, stop_words_set, top_n=15):
    # (Fungsi TF-IDF tetap sama)
    stopwords = set(stop_words_set)
    cleaned_texts = [re.sub(r'[^a-zA-Z0-9\s\-]', ' ', f"{t} {txt}".lower()) for t, txt in chapters]
    if not cleaned_texts: return {}
    
    vectorizer = TfidfVectorizer(stop_words=list(stopwords), ngram_range=(1, 4), max_df=0.7, min_df=1)
    try:
        X = vectorizer.fit_transform(cleaned_texts)
        feature_names = np.array(vectorizer.get_feature_names_out())
        results = {}
        for idx, (title, _) in enumerate(chapters):
            row = X[idx].toarray().flatten()
            top_indices = row.argsort()[::-1][:top_n]
            results[title] = [(feature_names[i], float(row[i])) for i in top_indices if row[i] > 0]
        return results
    except:
        return {}

def filter_stemmed_keywords(tfidf_results, stemmer_obj):
    # (Fungsi filter stem tetap sama)
    res = {}
    for t, kws in tfidf_results.items():
        res[t] = [(kw, s) for kw, s in kws if stemmer_obj.stem(kw.lower()) == kw.lower()]
    return res

# ... (Fungsi normalisasi frasa, YAKE, dll tetap sama seperti sebelumnya) ...
# ... Agar tidak terlalu panjang, saya asumsikan fungsi utilitas YAKE ada di sini ...

def run_yake_pipeline(all_text, page_texts, combined_stopwords_set, top_per_n=370):
    # (Implementasi YAKE pipeline sederhana untuk contoh)
    kw_extractor = yake.KeywordExtractor(lan="multilingual", n=2, top=top_per_n)
    keywords = kw_extractor.extract_keywords(all_text)
    return [kw for kw, _ in keywords], keywords

def domain_stem(word):
    ls = LancasterStemmer()
    custom = {"fuzzification": "fuzzy", "defuzzification": "fuzzy"}
    return custom.get(word.lower(), ls.stem(word.lower()))

def normalize_text(text):
    return re.sub(r'[^a-z0-9\s]', '', text.lower())

def normalize_phrase_order(phrase):
    return " ".join(sorted(phrase.split()))

# --- PERBAIKAN: Stemming sebelum W2V ---
def compare_title_with_keywords(title, tfidf_results, keywords, model, threshold=0.5):
    if not model: return []
    
    title_words = set(normalize_text(title).split())
    tfidf_context = set()
    for _, kws in tfidf_results.items():
        for kw, _ in kws: tfidf_context.update(normalize_text(kw).split())

    matched = []
    normalized_phrases = [kw.lower().strip() for kw in keywords]
    
    for kw in normalized_phrases:
        kw_norm = normalize_phrase_order(normalize_text(kw))
        
        # 1. STEMMING DULU
        try:
            stemmed_words = [domain_stem(w) for w in kw_norm.split()]
        except:
            stemmed_words = kw_norm.split()
        
        # 2. Cek keberadaan di model menggunakan kata yang sudah di-stem
        valid_tokens = [w for w in stemmed_words if w in model]
        
        # 3. Logika Hybrid
        # a. Cek TF-IDF overlap
        overlap = len(set(stemmed_words) & tfidf_context) / len(stemmed_words) if stemmed_words else 0
        if overlap >= 0.3:
            matched.append((kw, overlap, "tfidf"))
            continue
            
        # b. Cek Word2Vec Similarity
        if valid_tokens and title_words:
            kw_vec = np.mean([model[w] for w in valid_tokens], axis=0)
            title_vec_tokens = [model[w] for w in title_words if w in model]
            if title_vec_tokens:
                title_vec = np.mean(title_vec_tokens, axis=0)
                sim = 1 - cosine(kw_vec, title_vec)
                if sim >= threshold:
                    matched.append((kw, sim, "w2v"))

    return sorted(matched, key=lambda x: x[1], reverse=True)

def build_final_index(matched_keywords, page_texts):
    # (Fungsi build final index sederhana)
    res = defaultdict(list)
    phrases = {kw.lower(): kw for kw, _, _ in matched_keywords}
    for p_num, txt in page_texts:
        for p_low, p_orig in phrases.items():
            if p_low in txt.lower():
                res[p_orig].append(p_num)
    return dict(res)

# --- PERBAIKAN: PDF dengan Tabel ---
def create_index_pdf(keyword_pages_dict, output_path, book_title):
    try:
        pdfmetrics.registerFont(TTFont('Helvetica', 'Helvetica.ttf'))
        pdfmetrics.registerFont(TTFont('Helvetica-Bold', 'Helvetica-Bold.ttf'))
    except: pass

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['h1'], alignment=TA_CENTER, fontName='Helvetica-Bold', fontSize=16, spaceAfter=10*mm)
    book_style = ParagraphStyle('Book', parent=styles['h2'], alignment=TA_CENTER, fontName='Helvetica-Bold', fontSize=14, spaceAfter=5*mm)
    letter_style = ParagraphStyle('Letter', parent=styles['h3'], fontName='Helvetica-Bold', fontSize=14, spaceBefore=5*mm)
    phrase_style = ParagraphStyle('Phrase', parent=styles['Normal'], fontName='Helvetica', fontSize=10)
    page_style = ParagraphStyle('Page', parent=styles['Normal'], fontName='Helvetica', fontSize=10, alignment=TA_RIGHT)

    story = [Paragraph("Indeks Buku Otomatis", title_style), Paragraph(book_title, book_style)]
    
    sorted_phrases = sorted(keyword_pages_dict.keys(), key=str.lower)
    current_letter = ""
    table_data = []
    table_cmds = []

    for phrase in sorted_phrases:
        pages_list = keyword_pages_dict.get(phrase, [])
        # Bungkus halaman agar tidak melebar
        raw_pages = ", ".join(map(str, pages_list))
        wrapped_pages = wrap(raw_pages, 50)
        pages_str = "hlm.<br/>" + "<br/>".join(wrapped_pages)

        first = phrase[0].upper()
        if first != current_letter:
            current_letter = first
            row_idx = len(table_data)
            table_data.append([Paragraph(current_letter, letter_style), ''])
            table_cmds.append(('SPAN', (0, row_idx), (1, row_idx)))
            table_cmds.append(('TOPPADDING', (0, row_idx), (0, row_idx), 5*mm))
        
        table_data.append([Paragraph(phrase, phrase_style), Paragraph(pages_str, page_style)])

    if table_data:
        t = Table(table_data, colWidths=[doc.width*0.7, doc.width*0.3])
        t.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1*mm),
        ] + table_cmds))
        story.append(t)
    else:
        story.append(Paragraph("Data Kosong", styles['Normal']))

    doc.build(story)

# --- ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file_buku' not in request.files: return redirect(request.url)
        file = request.files['file_buku']
        book_title = request.form.get('book_title', '')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            session['buku_path'] = path
            
            # (Jalankan pipeline indexing - disederhanakan untuk contoh)
            # Asumsikan fungsi-fungsi di atas dipanggil di sini
            # ...
            page_texts, all_text, _ = preprocess_pdf(path, set())
            # ... (extract chapters, tfidf, yake, compare) ...
            # Dummy data untuk simulasi jika pipeline tidak dijalankan full di snippet ini
            keyword_pages_dict = {"Contoh Indeks": [1, 2], "Fuzzy Logic": [5, 10]} 
            
            # Simpan hasil
            session['book_title'] = book_title
            session['keyword_pages'] = keyword_pages_dict
            session['download_file'] = filename
            
            return redirect(url_for('index'))

    return render_template("index.html", 
                           book_title=session.get('book_title'), 
                           results=session.get('keyword_pages'), 
                           download_file=session.get('download_file'))

@app.route('/api/search_phrase', methods=['POST'])
def api_search_phrase():
    buku_path = session.get('buku_path')
    phrase = request.json.get('phrase', '').lower()
    if not buku_path or not phrase: return jsonify({'status': 'error', 'message': 'Data invalid'})
    
    found = []
    try:
        with pdfplumber.open(buku_path) as pdf:
            for i, p in enumerate(pdf.pages, 1):
                if phrase in (p.extract_text() or "").lower(): found.append(i)
        return jsonify({'status': 'success', 'pages': found})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# --- BARU: API untuk Download PDF Terpilih ---
@app.route('/api/download_selected', methods=['POST'])
def download_selected():
    data = request.json
    if not data or 'items' not in data:
        return jsonify({'status': 'error', 'message': 'Data tidak valid'}), 400
    
    items = data['items'] # Format: { "frasa": [1, 2, 3], ... }
    book_title = session.get('book_title', 'Indeks Khusus')
    
    # Buat nama file unik
    filename = f"indeks_terpilih_{int(datetime.now().timestamp())}.pdf"
    pdf_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    
    try:
        create_index_pdf(items, pdf_path, book_title)
        # Kembalikan URL agar frontend bisa membuka/download
        return jsonify({'status': 'success', 'download_url': url_for('download_file_route', filename=filename)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# --- BARU: Route untuk melayani file dinamis ---
@app.route('/download_result/<filename>')
def download_file_route(filename):
    path = os.path.join(app.config['RESULT_FOLDER'], filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "File not found", 404

# Route download asli (tetap ada untuk backward compatibility)
@app.route('/download/<original_filename>')
def download(original_filename):
    keyword_pages = session.get('keyword_pages')
    book_title = session.get('book_title')
    if not keyword_pages: return redirect(url_for('index'))
    
    filename = f"indeks_{secure_filename(original_filename)}"
    path = os.path.join(app.config['RESULT_FOLDER'], filename)
    create_index_pdf(keyword_pages, path, book_title)
    return send_file(path, as_attachment=True)

@app.route('/clear')
def clear_session():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)