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
import yake
from rapidfuzz import fuzz
from scipy.spatial.distance import cosine

# --- Import FastText ---
import fasttext
import fasttext.util

# --- Import Pembuatan PDF ---
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
stemmer_factory_global = StemmerFactory()
indonesian_stemmer = stemmer_factory_global.create_stemmer()

print("Memuat model FastText (mungkin perlu beberapa saat)...")
model_path_fasttext = r'C:\SKRIPSI (code)\models\cc.id.300.bin'

ft_model = None
try:
    if os.path.exists(model_path_fasttext):
        print(f"Memuat model FastText dari {model_path_fasttext}...")
        ft_model = fasttext.load_model(model_path_fasttext)
        print(f"Model FastText berhasil dimuat. Dimensi: {ft_model.get_dimension()}")
    else:
        print(f"PERINGATAN: File model FastText ({model_path_fasttext}) tidak ditemukan.")
        print("Fungsi yang bergantung pada embedding mungkin gagal.")
except Exception as e:
    print(f"Gagal memuat model FastText: {e}")

print("Model dan stopwords berhasil dimuat. Aplikasi siap.")


# --- Fungsi Bantuan ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# =============================================================================
# FUNGSI PREPROCESSING DAN EKSTRAKSI
# =============================================================================

def clean_text(text):
    """Membersihkan teks dari elemen yang tidak diinginkan."""
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
    """Normalisasi baris untuk deteksi header/footer."""
    line = line.lower().strip()
    line = re.sub(r"\d+", "", line)
    line = re.sub(r"\s+", " ", line)
    return line


def preprocess_pdf(pdf_path, dynamic_stopwords):
    """
    Ekstrak teks dari PDF, bersihkan header/footer, dan kembalikan teks per halaman.
    """
    header_even, header_odd, footer_all = [], [], []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                return [], "", "PDF kosong atau tidak bisa dibaca."

            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    lines = text.split("\n")
                    if len(lines) > 2:
                        header_line = normalize_line(lines[0])
                        footer_line = normalize_line(lines[-1])

                        if i % 2 == 0:
                            header_even.append(header_line)
                        else:
                            header_odd.append(header_line)

                        footer_all.append(footer_line)

        even_freq = Counter(header_even)
        odd_freq = Counter(header_odd)
        footer_freq = Counter(footer_all)

        n_even = len(header_even) if header_even else 1
        n_odd = len(header_odd) if header_odd else 1
        n_footer = len(footer_all) if footer_all else 1

        common_even = {h for h, c in even_freq.items() if (c / n_even > 0.3) and h}
        common_odd = {h for h, c in odd_freq.items() if (c / n_odd > 0.3) and h}
        common_footer = {f for f, c in footer_freq.items() if (c / n_footer > 0.3) and f}

        print(f"Header umum (genap): {common_even}")
        print(f"Header umum (ganjil): {common_odd}")
        print(f"Footer umum: {common_footer}")

        page_texts = []
        all_text = ""

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text()
                if raw_text:
                    lines = raw_text.split("\n")

                    if len(lines) > 2:
                        header = normalize_line(lines[0])
                        footer = normalize_line(lines[-1])

                        if i % 2 == 0 and header in common_even:
                            lines = lines[1:]
                        if i % 2 == 1 and header in common_odd:
                            lines = lines[1:]
                        if footer in common_footer:
                            lines = lines[:-1]

                    raw_text = "\n".join(lines)
                    clean = clean_text(raw_text)

                    # Tidak hapus stopwords di sini (sama seperti ipynb: "jika tidak")
                    page_texts.append((i, clean))
                    all_text += " " + clean

        return page_texts, all_text.strip(), None

    except Exception as e:
        return [], "", f"Error membaca PDF: {str(e)}"


# =============================================================================
# FUNGSI FASTTEXT EMBEDDING
# =============================================================================

def normalize_text(text):
    """Normalisasi teks untuk embedding FastText."""
    return re.sub(r'[^a-z0-9\s]', ' ', text.lower()).strip()


def phrase_embedding_fasttext(phrase, ft_model):
    """
    Menghasilkan vektor embedding untuk frasa menggunakan FastText.
    Frasa multi-kata dihitung sebagai rata-rata vektor kata penyusunnya.
    """
    if ft_model is None:
        return None

    tokens = normalize_text(phrase).split()
    if not tokens:
        return None

    vecs = [ft_model.get_word_vector(token) for token in tokens]
    return np.mean(vecs, axis=0)


# =============================================================================
# FUNGSI EKSTRAKSI DAN EKSPANSI AKRONIM
# =============================================================================

def extract_acronyms(text):
    """
    Ekstrak pemetaan akronim dari teks.
    Contoh: "Weighted Sum Model (WSM)" -> {'wsm': 'weighted sum model'}
    """
    pattern = r'([A-Za-z][A-Za-z\s\-]+)\s*\(([A-Z]{2,})\)'
    matches = re.findall(pattern, text)

    acronym_map = {}
    for long_form, short in matches:
        acronym_map[short.lower()] = long_form.lower().strip()

    return acronym_map


# =============================================================================
# FUNGSI PEMETAAN HALAMAN
# =============================================================================

def tokenize3(text):
    """Tokenisasi sederhana untuk pencocokan halaman."""
    return set(re.findall(r"\w+", text.lower()))


def map_keywords_to_pages(matched_keywords, page_texts, overlap_threshold=0.8):
    """
    Pemetaan keyword ke halaman-halaman kemunculannya.

    Args:
        matched_keywords: List dict {'keyword_full': ..., 'score': ..., ...}
        page_texts: List tuple (page_num, page_text)
        overlap_threshold: Threshold overlap minimum (default 0.8)

    Returns:
        dict: keyword -> list nomor halaman
    """
    matched_phrases = [item['keyword_full'] for item in matched_keywords]
    keyword_pages = defaultdict(list)

    for phrase in matched_phrases:
        phrase_tokens = tokenize3(phrase)
        if not phrase_tokens:
            continue

        for page_num, page_txt in page_texts:
            page_tokens = tokenize3(page_txt)
            if not page_tokens:
                continue

            overlap = len(phrase_tokens & page_tokens) / len(phrase_tokens)

            if overlap >= overlap_threshold or phrase.lower() in page_txt.lower():
                if page_num not in keyword_pages[phrase]:
                    keyword_pages[phrase].append(page_num)

    for phrase in keyword_pages:
        keyword_pages[phrase].sort()

    return dict(keyword_pages)


# =============================================================================
# FUNGSI YAKE PIPELINE - DISEDERHANAKAN SESUAI IPYNB
# =============================================================================

def filter_multiword_capitalized_phrases(keywords):
    """
    Named entity filtering: Pertahankan frasa multi-kata yang ada kapitalisasi.
    Contoh: 'Fuzzy Logic System' → pertahankan, 'fuzzy logic system' → hapus (jika multi-kata)
    """
    filtered = []
    for kw, score in keywords:
        words = kw.split()
        if len(words) > 1:
            if any(w and w[0].isupper() for w in words):
                filtered.append((kw, score))
        else:
            filtered.append((kw, score))
    return filtered


def filter_phrases_with_existing_acronyms(keywords):
    """
    Redundancy removal: Hapus frasa multi-kata yang mengandung akronim
    (di posisi manapun) jika akronim tersebut sudah ada sebagai keyword tersendiri.
    Contoh: Jika 'AHP' ada → hapus 'metode AHP', 'AHP Metode', 'metode AHP terbaik'
    """
    # Kumpulkan semua akronim yang berdiri sendiri (1 kata, semua huruf kapital)
    single_upper_keywords = {
        kw.upper() for kw, _ in keywords
        if len(kw.split()) == 1 and kw.isupper()
    }

    filtered = []
    for kw, score in keywords:
        words = kw.split()
        if len(words) > 1:
            # Frasa yang SEMUA katanya huruf kapital (misal "AHP WSM") → keep
            all_upper = all(w.isupper() for w in words)
            if all_upper:
                filtered.append((kw, score))
                continue

            # Hapus jika ada kata apapun dalam frasa yang merupakan akronim yang sudah ada
            # (posisi tidak dibatasi: awal, tengah, maupun akhir)
            upper_words = [w.upper() for w in words]
            should_remove = any(u in single_upper_keywords for u in upper_words)
            if should_remove:
                continue

        filtered.append((kw, score))
    return filtered


def normalize_repeated_words_second(phrases):
    """Token deduplication: Hilangkan kata berulang dalam frasa."""
    cleaned = []
    for ph in phrases:
        words = ph.split()
        new_words = []
        for w in words:
            if not new_words or w != new_words[-1]:
                new_words.append(w)
        unique_order = []
        seen = set()
        for w in new_words:
            if w.lower() not in seen:
                seen.add(w.lower())
                unique_order.append(w)
        cleaned.append(" ".join(unique_order))
    return cleaned


def merge_reversed_phrases_second(phrases):
    """Canonical form deduplication: Gabung frasa dengan set kata sama."""
    normalized = {}
    for ph in phrases:
        words = ph.split()
        key = tuple(sorted(words))
        if key not in normalized:
            normalized[key] = ph
    return list(normalized.values())


def is_base_word(word):
    """
    Cek apakah kata adalah kata dasar (bukan turunan).
    Return True jika kata dasar (keep), False jika turunan (hapus).
    """
    word_lower = word.lower().strip()

    # Kata pendek (<= 3 huruf) biasanya kata dasar
    if len(word_lower) <= 3:
        return True

    stemmed = indonesian_stemmer.stem(word_lower)

    # Jika hasil stem berbeda, kata ini adalah turunan
    if stemmed != word_lower:
        return False

    return True


def filter_single_word_derivatives(keyphrases):
    """
    Filter keyword 1 kata yang merupakan kata turunan (afiks).

    Logika:
    - Multi-word phrases: keep semua
    - Single-word: keep hanya jika kata dasar (hasil stemming == kata asli)

    Parameters:
    - keyphrases: List of keyword strings

    Returns:
    - filtered: List of keyword strings (setelah filtering)
    """
    filtered = []

    for phrase in keyphrases:
        words = phrase.split()

        # Multi-word phrase: selalu keep
        if len(words) > 1:
            filtered.append(phrase)
        else:
            # Single word: cek apakah kata dasar
            word = words[0]
            if is_base_word(word):
                filtered.append(phrase)
            # else: kata turunan → hapus (tidak ditambahkan)

    return filtered


def run_yake_pipeline(all_text, combined_stopwords_set, top_per_n=450):
    """
    Pipeline ekstraksi kata kunci YAKE yang disederhanakan sesuai ipynb.

    Filter yang digunakan (sesuai ipynb):
    1. Filter stopword & frasa semua kata sama
    2. filter_multiword_capitalized_phrases
    3. filter_phrases_with_existing_acronyms
    4. Lowercase seluruh keyphrases
    5. normalize_repeated_words_second
    6. merge_reversed_phrases_second
    7. filter_single_word_derivatives  ← tambahan dari ipynb

    Filter yang DIHAPUS dibanding auto_indexing.py:
    - filter_by_capitalization
    - boost_full_phrases_from_acronyms_v2
    - filter_similar_phrases_by_overlap_safe
    - merge_related_phrases (phrase consolidation berbasis halaman)

    Args:
        all_text: String teks lengkap buku
        combined_stopwords_set: Set stopwords
        top_per_n: Jumlah top keyword per n-gram (default 450, sama dengan ipynb)

    Returns:
        tuple: (keyphrases_final_strings, cleaned_keywords_tuples)
    """
    print("\n" + "=" * 80)
    print("MEMULAI YAKE PIPELINE (v2 - sesuai ipynb)")
    print("=" * 80)

    # 1. YAKE Extraction (n=1,2,3 — sama dengan ipynb)
    keywords_all = []
    for n in [1, 2, 3]:
        kw_extractor = yake.KeywordExtractor(lan="multilingual", n=n, top=top_per_n)
        keywords = kw_extractor.extract_keywords(all_text)
        keywords_all.extend(keywords)

    # 2. Initial Cleaning: deduplikasi, filter stopword, frasa semua kata sama
    seen = set()
    cleaned_keywords = []
    for kw, score in sorted(keywords_all, key=lambda x: x[1]):
        kw = kw.strip()
        if not kw:
            continue
        words = kw.split()

        # Buang frasa semua kata sama (misal "model model")
        if len(words) > 1 and len(set(words)) == 1:
            continue

        # Buang jika ada kata yang masuk stopwords
        if any(w.lower() in combined_stopwords_set for w in words):
            continue

        norm_kw = kw.lower()
        if norm_kw not in seen:
            seen.add(norm_kw)
            cleaned_keywords.append((kw, score))

    cleaned_keywords = sorted(cleaned_keywords, key=lambda x: x[1])
    print(f"✓ YAKE (raw): {len(cleaned_keywords)} keywords")

    # 3. Filter frasa multi-kata non-kapital
    capital_phrase_filtered = filter_multiword_capitalized_phrases(cleaned_keywords)
    print(f"✓ After filter non-capitalized multi-word: {len(capital_phrase_filtered)} keywords")

    # 4. Buang frasa yang mengandung akronim yang sudah ada
    acronym_filtered = filter_phrases_with_existing_acronyms(capital_phrase_filtered)
    print(f"✓ After filter by acronym: {len(acronym_filtered)} keywords")

    # 5. Lowercase semua keyphrases
    keyphrases = [kw.lower() for kw, _ in acronym_filtered]

    # 6. Normalisasi dan deduplication akhir
    keyphrases = normalize_repeated_words_second(keyphrases)
    keyphrases = merge_reversed_phrases_second(keyphrases)
    print(f"✓ After normalize & merge reversed: {len(keyphrases)} keywords")

    # 7. Filter kata tunggal yang merupakan kata turunan (sesuai ipynb)
    keyphrases = filter_single_word_derivatives(keyphrases)
    print(f"✓ After filter single-word derivatives: {len(keyphrases)} keywords")

    print("=" * 80 + "\n")

    return keyphrases, cleaned_keywords


# =============================================================================
# FUNGSI EKSTRAKSI KEYWORD DARI KONTEKS (JUDUL + ABSTRAK)
# =============================================================================

def extract_context_keywords(book_title, book_summary, combined_stopwords_set, top_per_n=350):
    """
    Ekstrak keyword dari judul + abstrak menggunakan YAKE,
    sehingga perbandingan FastText dilakukan frasa-vs-frasa (bukan teks-vs-frasa).

    Ini adalah perubahan utama dari auto_indexing.py:
    - Sebelumnya: judul+abstrak digabung → satu embedding vektor konteks
    - Sekarang (v2): judul+abstrak di-ekstrak keyword-nya → list frasa konteks
      yang kemudian dibandingkan frasa-per-frasa dengan keyword buku

    Args:
        book_title: String judul buku
        book_summary: String abstrak/ringkasan buku
        combined_stopwords_set: Set stopwords
        top_per_n: Jumlah top keyword YAKE untuk konteks

    Returns:
        list: List string keyword konteks yang sudah bersih
    """
    context_text = f"{book_title} {book_summary}"

    keywords_all = []
    for n in [1, 2, 3]:
        kw_extractor = yake.KeywordExtractor(lan="multilingual", n=n, top=top_per_n)
        keywords = kw_extractor.extract_keywords(context_text)
        keywords_all.extend(keywords)

    seen = set()
    context_keywords = []
    for kw, score in sorted(keywords_all, key=lambda x: x[1]):
        kw = kw.strip().lower()
        if not kw:
            continue
        words = kw.split()

        if len(words) > 1 and len(set(words)) == 1:
            continue
        if any(w in combined_stopwords_set for w in words):
            continue

        if kw not in seen:
            seen.add(kw)
            context_keywords.append(kw)

    print(f"✓ Context keywords dari judul+abstrak: {len(context_keywords)} frasa")
    return context_keywords


# =============================================================================
# FUNGSI PEMFILTERAN BERBASIS KONTEKS (FASTTEXT) - FRASA VS FRASA
# =============================================================================

def compare_keywords_fasttext(keywords_full, context_keywords_clean, ft_model, threshold=0.5):
    """
    Bandingkan keyword buku dengan keyword konteks (judul+abstrak) menggunakan FastText.

    Perubahan dari auto_indexing.py:
    - Sebelumnya: setiap keyword buku dibandingkan dengan SATU embedding konteks gabungan
    - Sekarang (v2): setiap keyword buku dibandingkan dengan SETIAP frasa konteks,
      diambil similarity tertinggi → frasa-vs-frasa (sesuai ipynb)

    Args:
        keywords_full: List string keyword hasil YAKE dari teks buku
        context_keywords_clean: List string keyword dari judul+abstrak
        ft_model: Model FastText
        threshold: Threshold similarity minimum (default 0.5)

    Returns:
        List dict: [{'keyword_full', 'best_context_match', 'score', 'method'}, ...]
        Diurutkan dari skor tertinggi.
    """
    if ft_model is None:
        print("PERINGATAN: Model FastText tidak tersedia. Mengembalikan semua keyword.")
        return [
            {'keyword_full': kw, 'best_context_match': None, 'score': 1.0, 'method': 'no-filter'}
            for kw in keywords_full
        ]

    # 1. Hitung embedding semua frasa konteks sekali di awal (efisiensi)
    ctx_data = []
    for kw_ctx in context_keywords_clean:
        ctx_text = normalize_text(kw_ctx)
        if not ctx_text:
            continue
        vec_ctx = phrase_embedding_fasttext(ctx_text, ft_model)
        if vec_ctx is not None:
            ctx_data.append((ctx_text, vec_ctx))

    if not ctx_data:
        print("PERINGATAN: Tidak ada frasa konteks yang valid. Mengembalikan semua keyword.")
        return [
            {'keyword_full': kw, 'best_context_match': None, 'score': 1.0, 'method': 'no-filter'}
            for kw in keywords_full
        ]

    # 2. Bandingkan setiap keyword buku dengan semua frasa konteks
    matched = []
    for kw_full in keywords_full:
        full_text = normalize_text(kw_full)
        if not full_text:
            continue

        vec_full = phrase_embedding_fasttext(full_text, ft_model)
        if vec_full is None:
            continue

        best_sim = -1
        best_match = None

        for ctx_text, vec_ctx in ctx_data:
            # Exact match → skor otomatis 1.0
            if full_text == ctx_text:
                sim = 1.0
            else:
                sim = 1 - cosine(vec_full, vec_ctx)

            if sim > best_sim:
                best_sim = sim
                best_match = ctx_text

        # Masukkan jika memenuhi threshold
        if best_sim >= threshold:
            matched.append({
                'keyword_full': full_text,
                'best_context_match': best_match,
                'score': round(float(best_sim), 4),
                'method': 'fasttext-context'
            })

    # Urutkan dari skor tertinggi
    return sorted(matched, key=lambda x: x['score'], reverse=True)


# =============================================================================
# FUNGSI EVALUASI
# =============================================================================

def fasttext_embed(text, ft_model):
    """Wrapper untuk phrase_embedding_fasttext dengan handling None."""
    tokens = text.lower().split()
    if not tokens or ft_model is None:
        return np.zeros(ft_model.get_dimension() if ft_model else 300)

    vecs = [ft_model.get_word_vector(t) for t in tokens]
    return np.mean(vecs, axis=0)


def average_fasttext_similarity(set1, set2, ft_model):
    """
    Menghitung rata-rata similarity antara dua set frasa menggunakan FastText.
    """
    if not set1 or not set2 or ft_model is None:
        return 0.0

    from sklearn.metrics.pairwise import cosine_similarity

    emb1 = np.array([fasttext_embed(p, ft_model) for p in set1])
    emb2 = np.array([fasttext_embed(p, ft_model) for p in set2])

    sims = cosine_similarity(emb1, emb2)
    best_sims = sims.max(axis=1)

    return float(best_sims.mean())


def preprocess_phrases(phrases):
    """Preprocessing frasa untuk evaluasi."""
    roman_numerals = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}
    cleaned = set()

    for phrase in phrases:
        phrase = re.sub(r"\(.*?\)", "", phrase)
        phrase = re.sub(r"[^\w\s]", " ", phrase)
        phrase = re.sub(r"\s+", " ", phrase).lower().strip()

        if (
            any(c.isalpha() for c in phrase)
            and len(phrase) > 1
            and phrase not in roman_numerals
            and not re.fullmatch(r"[a-zA-Z]", phrase)
        ):
            cleaned.add(phrase)

    return cleaned


def extract_keywords_from_index_file(gt_path):
    """
    Ekstrak keyword dari file indeks ground truth (PDF).
    """
    doc = fitz.open(gt_path)
    keywords = set()

    for page in doc:
        text = page.get_text("text")
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        merged_lines = []
        buffer = ""

        def is_complete_entry(line):
            return bool(re.search(r'[·,]\s*[ivxIVX\d]', line))

        def is_new_entry(line):
            return bool(re.match(r'^[A-Z]', line))

        for line in lines:
            if re.match(r'^[A-Z]$', line):
                if buffer:
                    merged_lines.append(buffer)
                    buffer = ""
                continue

            if buffer:
                if is_new_entry(line) and is_complete_entry(buffer):
                    merged_lines.append(buffer)
                    buffer = line
                else:
                    buffer += " " + line
                    if is_complete_entry(buffer):
                        merged_lines.append(buffer)
                        buffer = ""
            else:
                if is_complete_entry(line):
                    merged_lines.append(line)
                else:
                    buffer = line

        if buffer:
            merged_lines.append(buffer)

        for line in merged_lines:
            match = re.match(r'^([A-Za-z][A-Za-z\s\-\(\)]*?)\s*[·,]', line.strip())
            if match:
                kw = match.group(1).strip()
                if len(kw) > 1:
                    keywords.add(kw)

    return preprocess_phrases(keywords)


def fuzzy_match_evaluation(generated_keywords, ground_truth_keywords, threshold=85):
    """
    Evaluasi menggunakan fuzzy matching untuk mengatasi typo.
    """
    tp = set()
    fp = set()
    matched_gt = set()

    for gen_kw in generated_keywords:
        best_match = None
        best_score = 0

        for gt_kw in ground_truth_keywords:
            if gt_kw in matched_gt:
                continue

            if len(gen_kw) <= 3:
                adaptive_threshold = 60
            elif len(gen_kw) <= 4:
                adaptive_threshold = 75
            else:
                adaptive_threshold = threshold

            score = fuzz.ratio(gen_kw.lower(), gt_kw.lower())

            if score >= adaptive_threshold and score > best_score:
                best_score = score
                best_match = gt_kw

        if best_match:
            tp.add(gen_kw)
            matched_gt.add(best_match)
        else:
            fp.add(gen_kw)

    fn = ground_truth_keywords - matched_gt

    return tp, fp, fn


def evaluasi_indeks(gt_path, matched_keywords, ft_model=None, use_fuzzy=True):
    """
    Evaluasi hasil indeks buku terhadap indeks ground truth.

    Args:
        gt_path: Path ke file ground truth (PDF)
        matched_keywords: List dict {'keyword_full': ..., 'score': ..., ...}
        ft_model: Model FastText untuk similarity
        use_fuzzy: Boolean, gunakan fuzzy matching atau tidak

    Returns:
        dict: Hasil evaluasi (precision, recall, f1, similarity, dll)
    """
    generated_keywords = preprocess_phrases([item['keyword_full'] for item in matched_keywords])
    ground_truth_keywords = extract_keywords_from_index_file(gt_path)

    if use_fuzzy:
        print("Menggunakan fuzzy matching untuk evaluasi...")
        tp, fp, fn = fuzzy_match_evaluation(generated_keywords, ground_truth_keywords)
    else:
        print("Menggunakan exact matching untuk evaluasi...")
        tp = generated_keywords & ground_truth_keywords
        fp = generated_keywords - ground_truth_keywords
        fn = ground_truth_keywords - generated_keywords

    precision = len(tp) / len(generated_keywords) if generated_keywords else 0
    recall = len(tp) / len(ground_truth_keywords) if ground_truth_keywords else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    ft_sim = None
    if ft_model is not None:
        ft_sim = average_fasttext_similarity(generated_keywords, ground_truth_keywords, ft_model)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "similarity": round(ft_sim, 4) if ft_sim is not None else None,
        "true_positives": sorted(tp),
        "false_positives": sorted(fp),
        "false_negatives": sorted(fn),
        "fuzzy_matching_used": use_fuzzy
    }


# =============================================================================
# FUNGSI PEMBUATAN PDF INDEKS
# =============================================================================

def create_index_pdf(keyword_pages, pdf_path, book_title):
    """
    Membuat file PDF indeks buku.
    """
    try:
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER
        )

        story.append(Paragraph(f"Indeks Buku: {book_title}", title_style))
        story.append(Spacer(1, 12))

        data = [["Kata Kunci", "Halaman"]]

        for keyword in sorted(keyword_pages.keys()):
            pages = keyword_pages[keyword]
            pages_str = ", ".join(map(str, pages))
            data.append([keyword, pages_str])

        table = Table(data, colWidths=[300, 200])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(table)
        doc.build(story)

        print(f"PDF indeks berhasil dibuat: {pdf_path}")

    except Exception as e:
        print(f"Error membuat PDF: {e}")
        raise


# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    """Route utama untuk tampilan aplikasi."""
    keyword_pages = session.get('keyword_pages')
    book_title = session.get('book_title')
    book_summary = session.get('book_summary')
    download_file = session.get('download_file')
    eval_results = session.get('eval_results')

    return render_template(
        'index.html',
        keyword_pages=keyword_pages,
        book_title=book_title,
        book_summary=book_summary,
        download_file=download_file,
        eval_results=eval_results
    )


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Route untuk upload dan proses file buku.
    Menerima input: file PDF, judul buku, ringkasan buku, dan stopwords tambahan.
    """
    if 'file_buku' not in request.files:
        flash('Tidak ada file yang diunggah', 'error')
        return redirect(url_for('index'))

    file = request.files['file_buku']
    if file.filename == '':
        flash('Tidak ada file yang dipilih', 'error')
        return redirect(url_for('index'))

    book_title = request.form.get('book_title', '').strip()
    book_summary = request.form.get('book_summary', '').strip()

    if not book_title:
        flash('Judul buku harus diisi', 'error')
        return redirect(url_for('index'))

    if not book_summary:
        flash('Ringkasan buku harus diisi', 'error')
        return redirect(url_for('index'))

    if len(book_summary) < 50:
        flash('Ringkasan buku minimal 50 karakter', 'error')
        return redirect(url_for('index'))

    extra_stopwords_input = request.form.get('extra_stopwords', '').strip()
    extra_stopwords = set()
    if extra_stopwords_input:
        extra_stopwords = {w.strip().lower() for w in extra_stopwords_input.split(',') if w.strip()}

    dynamic_stopwords = combined_stopwords.union(extra_stopwords)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"File disimpan di: {filepath}")
        print(f"Judul buku: {book_title}")
        print(f"Ringkasan buku: {book_summary[:100]}...")
        print(f"Stopwords tambahan: {extra_stopwords}")

        try:
            # 1. Preprocessing PDF
            print("Memulai preprocessing PDF...")
            page_texts, all_text, error = preprocess_pdf(filepath, dynamic_stopwords)

            if error:
                flash(f'Error: {error}', 'error')
                return redirect(url_for('index'))

            print(f"Teks berhasil diekstrak: {len(all_text)} karakter")

            # 2. Ekstraksi kata kunci dari teks buku dengan YAKE
            print("Ekstraksi kata kunci buku dengan YAKE...")
            keyphrases, cleaned_keywords = run_yake_pipeline(
                all_text,
                dynamic_stopwords,
                top_per_n=450  # sesuai ipynb
            )
            print(f"Keyword hasil YAKE (buku): {len(keyphrases)}")

            # 3. Ekstraksi keyword dari konteks (judul + abstrak)
            #    PERUBAHAN dari v1: judul+abstrak diekstrak keyword-nya dulu,
            #    bukan langsung dijadikan satu embedding vektor.
            print("Mengekstrak keyword dari judul + abstrak...")
            context_keywords = extract_context_keywords(
                book_title,
                book_summary,
                dynamic_stopwords,
                top_per_n=350
            )

            # 4. Filter keyword buku berdasarkan konteks (frasa vs frasa)
            #    PERUBAHAN dari v1: perbandingan dilakukan frasa-vs-frasa,
            #    bukan frasa-vs-embedding-dokumen.
            print("Memfilter keyword buku terhadap frasa konteks dengan FastText...")
            matched_keywords = compare_keywords_fasttext(
                keywords_full=keyphrases,
                context_keywords_clean=context_keywords,
                ft_model=ft_model,
                threshold=0.5  # sesuai ipynb
            )
            print(f"Keyword yang match dengan konteks: {len(matched_keywords)}")

            # 5. Pemetaan halaman
            print("Memetakan keyword ke halaman...")
            keyword_pages = map_keywords_to_pages(matched_keywords, page_texts)
            print(f"Keyword dengan halaman: {len(keyword_pages)}")

            # 6. Simpan ke sesi
            session['keyword_pages'] = keyword_pages
            session['matched_keywords'] = matched_keywords
            session['book_title'] = book_title
            session['book_summary'] = book_summary
            session['buku_path'] = filepath
            session['download_file'] = filename

            print("Proses selesai. Data disimpan di sesi.")
            flash(f'✅ File berhasil diproses! Ditemukan {len(keyword_pages)} kata kunci.', 'success')

        except Exception as e:
            print(f"Error saat memproses file: {e}")
            import traceback
            traceback.print_exc()
            flash(f'Terjadi error: {str(e)}', 'error')
            return redirect(url_for('index'))
    else:
        flash('Format file tidak valid. Hanya PDF yang diperbolehkan.', 'error')

    return redirect(url_for('index'))


@app.route('/api/search_phrase', methods=['POST'])
def api_search_phrase():
    """
    API untuk mencari frasa di dalam buku dan mengembalikan nomor halaman.
    """
    buku_path = session.get('buku_path')
    if not buku_path or not os.path.exists(buku_path):
        return jsonify({'status': 'error', 'message': 'Sesi buku tidak ditemukan. Harap unggah ulang file buku.'}), 400

    data = request.json
    phrase = data.get('phrase', '').strip()
    if not phrase:
        return jsonify({'status': 'error', 'message': 'Frasa tidak boleh kosong.'}), 400

    found_pages = []
    try:
        with pdfplumber.open(buku_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and (phrase.lower() in text.lower()):
                    found_pages.append(i)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Gagal membaca PDF: {e}'}), 500

    if found_pages:
        return jsonify({'status': 'success', 'pages': found_pages})
    else:
        return jsonify({'status': 'not_found', 'message': 'Indeks tidak ada di buku ini.'})


@app.route('/api/bulk_delete', methods=['POST'])
def api_bulk_delete():
    """Menghapus item dari sesi berdasarkan list frasa."""
    try:
        data = request.json
        phrases_to_delete = data.get('phrases', [])

        current_index = session.get('keyword_pages', {})

        if not current_index:
            return jsonify({'status': 'error', 'message': 'Data sesi kosong.'}), 400

        deleted_count = 0
        for phrase in phrases_to_delete:
            if phrase in current_index:
                del current_index[phrase]
                deleted_count += 1

        session['keyword_pages'] = current_index
        session.modified = True

        return jsonify({'status': 'success', 'deleted_count': deleted_count})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/download_selected_pdf', methods=['POST'])
def api_download_selected_pdf():
    """Membuat dan mengirim PDF yang hanya berisi frasa yang dipilih user."""
    try:
        data = request.json
        selected_phrases = data.get('phrases', [])

        if not selected_phrases:
            return jsonify({'status': 'error', 'message': 'Tidak ada frasa yang dipilih.'}), 400

        full_index = session.get('keyword_pages', {})
        book_title = session.get('book_title', 'Indeks Terpilih')

        filtered_data = {}
        for phrase in selected_phrases:
            if phrase in full_index:
                filtered_data[phrase] = full_index[phrase]

        if not filtered_data:
            return jsonify({'status': 'error', 'message': 'Data terpilih tidak ditemukan di sesi.'}), 404

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"indeks_terpilih_{timestamp}.pdf"
        pdf_path = os.path.join(app.config['RESULT_FOLDER'], pdf_filename)

        create_index_pdf(filtered_data, pdf_path, f"{book_title} (Terpilih)")

        return send_file(pdf_path, as_attachment=True, download_name=pdf_filename)

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/evaluasi', methods=['POST'])
def evaluasi():
    """Route untuk Evaluasi F1 Score."""
    matched_keywords = session.get('matched_keywords')
    if not matched_keywords:
        flash('Sesi tidak ditemukan. Harap unggah file buku terlebih dahulu.', 'error')
        return redirect(url_for('index'))

    if 'file_gt' not in request.files:
        flash('Tidak ada file ground truth yang diunggah', 'error')
        return redirect(url_for('index'))

    file_gt = request.files['file_gt']
    if file_gt.filename == '':
        flash('Tidak ada file ground truth yang dipilih', 'error')
        return redirect(url_for('index'))

    if file_gt and allowed_file(file_gt.filename):
        gt_filename = f"gt_{secure_filename(file_gt.filename)}"
        gt_path = os.path.join(app.config['UPLOAD_FOLDER'], gt_filename)
        file_gt.save(gt_path)

        print(f"File Ground Truth disimpan di: {gt_path}")
        print("Memulai evaluasi...")

        if not ft_model:
            flash('Model FastText tidak dimuat, evaluasi similarity mungkin tidak akurat.', 'warning')

        use_fuzzy = request.form.get('use_fuzzy', 'true').lower() == 'true'

        eval_results = evaluasi_indeks(gt_path, matched_keywords, ft_model, use_fuzzy=use_fuzzy)
        print("Evaluasi selesai.")

        if eval_results.get('error'):
            flash(eval_results['error'], 'error')

        session['eval_results'] = eval_results

    return redirect(url_for('index'))


@app.route('/download/<original_filename>')
def download(original_filename):
    """Route untuk mengunduh hasil indeks (PDF)."""
    keyword_pages = session.get('keyword_pages')
    book_title = session.get('book_title')

    if not keyword_pages or not book_title:
        flash('Sesi hasil indeks tidak ditemukan. Harap proses ulang file buku.', 'error')
        return redirect(url_for('index'))

    pdf_filename = f"indeks_{secure_filename(original_filename)}"
    pdf_path = os.path.join(app.config['RESULT_FOLDER'], pdf_filename)

    try:
        create_index_pdf(keyword_pages, pdf_path, book_title)

        if os.path.exists(pdf_path):
            return send_file(pdf_path, as_attachment=True)
        else:
            flash('Gagal membuat file PDF.', 'error')
            return redirect(url_for('index'))

    except Exception as e:
        flash(f'Terjadi error saat membuat PDF: {e}', 'error')
        return redirect(url_for('index'))


@app.route('/clear')
def clear_session():
    """Membersihkan sesi dan memulai ulang."""
    session.clear()
    return redirect(url_for('index'))


# --- Jalankan Aplikasi ---
if __name__ == '__main__':
    if not os.path.exists(model_path_fasttext):
        print("-" * 70)
        print(f"PERINGATAN: File model FastText '{model_path_fasttext}' tidak ditemukan.")
        print(f"Harap unduh model dan letakkan di: {os.path.dirname(model_path_fasttext)}")
        print("Download dari: https://fasttext.cc/docs/en/crawl-vectors.html")
        print("Untuk bahasa Indonesia: cc.id.300.bin")
        print("-" * 70)

    app.run(debug=True)
