import os
import re
import math
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict

# --- Flask ---
from flask import Flask, render_template, request, send_file, session, redirect, url_for, flash, jsonify
from flask_session import Session
from werkzeug.utils import secure_filename

# --- PDF & Teks ---
import pdfplumber
import fitz  # PyMuPDF
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk
import yake
from rapidfuzz import fuzz
from scipy.spatial.distance import cosine

# --- FastText ---
import fasttext

# --- Summarizer (sumy) ---
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# --- PDF Output ---
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors

# =============================================================================
# KONFIGURASI FLASK
# =============================================================================

app = Flask(__name__)
app.secret_key = 'kunci-rahasia-auto-indexing-v3'

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
Session(app)

ALLOWED_EXTENSIONS = {'pdf'}

# =============================================================================
# STOPWORDS
# =============================================================================

# Stopwords manual Bahasa Indonesia — diperluas dari versi ipynb
# Mencakup kata berimbuhan, kata modalitas, kata prosedural,
# dan kata-kata teknis yang tidak bermakna sebagai entri indeks
STOPWORDS_ID = {
    # Konjungsi & preposisi dasar
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk",
    "dalam", "adalah", "pada", "atau", "tidak", "juga", "akan", "dapat",
    "oleh", "sebagai", "ada", "telah", "maka", "sehingga", "agar", "bagi",
    "karena", "seperti", "antara", "setiap", "suatu", "berdasarkan",
    "tersebut", "dimana", "bahwa", "adapun", "yaitu", "yakni", "masing",
    "tiap", "setelah", "sebelum", "kemudian", "selanjutnya", "pertama",
    "kedua", "ketiga", "langkah", "cara", "berikut", "berikutnya",
    "lebih", "sangat", "hanya", "jika", "apabila", "hingga", "sampai",
    "melalui", "terhadap", "jadi", "sudah", "pun", "hal", "namun",
    "tetapi", "tapi", "walau", "meski", "meskipun", "walaupun",
    "bisa", "boleh", "harus", "perlu", "ingin", "lain", "lainnya",
    "serta", "maupun", "baik", "bukan", "belum", "masih", "paling",
    # Kata proses/aktif yang tidak bermakna sebagai entri indeks
    "menggunakan", "digunakan", "dilakukan",
    "dihitung", "diperoleh", "didapatkan", "ditentukan", "dibuat",
    "terdiri", "terbentuk", "merupakan", "memiliki", "mempunyai",
    "dihasilkan", "diberikan", "disebut", "dikenal",
    # Angka & bilangan
    "salah", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh",
    "delapan", "sembilan", "sepuluh", "semua", "seluruh", "sebuah",
    "beberapa", "banyak",
    # Kata penunjuk dokumen
    "tabel", "gambar", "bab",
    "keterangan", "dst", "dll", "dsb", "lampiran", "hlm", "hal",
    "cet", "edisi", "cetakan", "penerbit", "pengarang", "penulis",
    # Angka romawi
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
    "xi", "xii",
    # ----------------------------------------------------------------
    # TAMBAHAN: kata berimbuhan yang tidak bermakna sebagai entri indeks
    # ----------------------------------------------------------------
    # Kata berimbuhan me-
    "melainkan", "melakukan", "membuat", "menjadi", "menyebabkan",
    "menjelaskan", "menunjukkan", "menghasilkan", "menentukan",
    "mendapatkan", "menghitung", "menggunakan", "memperoleh",
    "meningkatkan", "menurunkan", "mengurangi", "menambah",
    "membantu", "mencapai", "melihat", "mengetahui", "memahami",
    "membutuhkan", "memerlukan", "mempengaruhi", "menggambarkan",
    "menyatakan", "menerapkan", "memberikan", "melibatkan",
    "mengambil", "mengeluarkan", "menyusun", "membentuk",
    "mengandung", "mengubah", "menyediakan", "mengelola",
    "mengidentifikasi", "mengevaluasi", "menganalisis",
    # Kata berimbuhan di-
    "dikembangkan", "diterapkan", "dianalisis", "dievaluasi",
    "diidentifikasi", "dikelola", "diubah", "disusun", "dibentuk",
    "diambil", "disediakan", "digunakan", "diukur", "dipilih",
    "ditetapkan", "dijelaskan", "ditunjukkan", "digambarkan",
    "dinyatakan", "disebutkan", "dikelompokkan", "dibandingkan",
    "diperhitungkan", "dijadikan", "diasumsikan",
    # Kata berimbuhan ke-an / pe-an
    # "penggunaan", "penerapan", "pengembangan", "perhitungan",
    # "penentuan", "pembentukan", "penyusunan", "pengelolaan",
    # "penilaian", "pengukuran", "pemilihan", "pembahasan",
    # "perbandingan", "pendekatan", "penjelasan", "peningkatan",
    # "penurunan", "penambahan", "pengurangan", "pembuatan",
    # "pelaksanaan", "pelakukan", "pemanfaatan", "penanganan",
    # "perangkingan", "pengolahan",
    # Kata berimbuhan ter-
    "terbaik",
    "terpilih", "terakhir", "terbaru", "tertentu", "terdapat",
    "terlihat", "terutama", "tergantung", "termasuk", "terjadi",
    "terkait", "terkecuali", "terhadap",
    # Kata berimbuhan ber-
    "berbeda", "berkaitan", "berhubungan", "berpengaruh", "berupa",
    "berjumlah", "berada", "berlaku", "berfungsi", "berperan",
    "berkembang", "bertujuan", "bersifat", "berasal", "berdasar",
    "bergantung",
    # Kata modalitas & keterangan waktu
    "seharusnya", "sebaiknya", "kadang", "kadangkala", "sering",
    "selalu", "jarang", "biasanya", "umumnya", "akhirnya",
    "awalnya", "sebenarnya", "sesungguhnya", "tentunya", "tentu",
    "memang", "justru", "bahkan", "namun", "oleh karena",
    "setidaknya", "setidak", "sekurangnya", "paling tidak",
    "sekaligus", "sekarang", "saat ini", "saat",
    # Kata penjelas / prosa akademis
    "tersebut di atas", "berikut ini", "sebagai berikut",
    "antara lain", "dan lain", "dan sebagainya", "yaitu",
    "yakni", "misalnya", "contohnya", "artinya", "maksudnya",
    "dimaksud", "disini", "dimana", "sehingga", "supaya",
    # Kata evaluatif generik
    "baik", "buruk", "tinggi", "rendah", "besar", "kecil",
    "penting", "relevan", "efektif", "efisien", "optimal",
    "akurat", "valid", "reliabel", "objektif",
    # Kata struktural dokumen akademis
    "pendahuluan", "penutup", "kesimpulan", "saran", "daftar",
    "pustaka", "referensi", "abstrak", "kata pengantar",
    "daftar isi", "daftar tabel", "daftar gambar",
    # Kata umum yang sering muncul tapi tidak jadi entri indeks
    "tahap",
    "langkah", "aspek", "faktor",
    "jenis", "bentuk", "tipe", "kategori", "kelompok", "kelas",
    "tingkat", "ukuran", "jumlah", "angka", "data",
    "informasi", "pengetahuan",
    # "metode", "unsur", "komponen", "elemen", "skala", "konsisten", "bertambah", "berkurang",
    #  "tertinggi", "terendah", "terbesar", "terkecil", "sistem", "teknik", "proses", "prosedur",
    # "contoh", "kasus", "nilai", "skor", "hasil", "bagian", "persamaan", "rumus",
}

# =============================================================================
# INISIALISASI GLOBAL
# =============================================================================

print("Memuat stopwords...")
try:
    stopwords.words('indonesian')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

factory = StopWordRemoverFactory()
sastrawi_stop = set(factory.get_stop_words())
nltk_stop_id = set(stopwords.words("indonesian"))
nltk_stop_en = set(stopwords.words("english"))
combined_stopwords = sastrawi_stop.union(nltk_stop_id).union(nltk_stop_en).union(STOPWORDS_ID)

print("Memuat stemmer Sastrawi...")
stemmer_factory_global = StemmerFactory()
indonesian_stemmer = stemmer_factory_global.create_stemmer()

FASTTEXT_PATH = r'C:\SKRIPSI (code)\models\cc.id.300.bin'
ft_model = None
try:
    if os.path.exists(FASTTEXT_PATH):
        print(f"Memuat model FastText dari {FASTTEXT_PATH}...")
        ft_model = fasttext.load_model(FASTTEXT_PATH)
        print(f"Model FastText berhasil dimuat. Dimensi: {ft_model.get_dimension()}")
    else:
        print(f"PERINGATAN: Model FastText tidak ditemukan di {FASTTEXT_PATH}")
except Exception as e:
    print(f"Gagal memuat FastText: {e}")

print("Inisialisasi selesai.")


# =============================================================================
# FUNGSI BANTUAN
# =============================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# =============================================================================
# PREPROCESSING PDF
# =============================================================================

def normalize_line(line):
    line = line.lower().strip()
    line = re.sub(r"\d+", "", line)
    line = re.sub(r"\s+", " ", line)
    return line


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


def clean_rumus(text):
    """Membersihkan teks dari rumus matematika dan simbol."""
    text = re.sub(r'[𝑎-𝑧𝑨-𝒁𝟎-𝟗]', '', text)
    text = re.sub(r'^[0-9\.\,\=\-\+\*\/\(\) ]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s{0,3}[A-Za-z0-9]{1,3}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_pdf(pdf_path):
    """
    Ekstrak teks dari PDF, bersihkan header/footer berulang.
    Return: (page_texts, all_text, error)
    """
    header_even, header_odd, footer_all = [], [], []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                return [], "", "PDF kosong."

            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    lines = text.split("\n")
                    if len(lines) > 2:
                        h = normalize_line(lines[0])
                        f = normalize_line(lines[-1])
                        if i % 2 == 0:
                            header_even.append(h)
                        else:
                            header_odd.append(h)
                        footer_all.append(f)

        def common_set(lst, threshold=0.3):
            if not lst:
                return set()
            freq = Counter(lst)
            n = len(lst)
            return {h for h, c in freq.items() if c / n > threshold and h}

        common_even = common_set(header_even)
        common_odd = common_set(header_odd)
        common_footer = common_set(footer_all)

        page_texts = []
        all_text = ""

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text()
                if raw:
                    lines = raw.split("\n")
                    if len(lines) > 2:
                        h = normalize_line(lines[0])
                        f = normalize_line(lines[-1])
                        if i % 2 == 0 and h in common_even:
                            lines = lines[1:]
                        if i % 2 == 1 and h in common_odd:
                            lines = lines[1:]
                        if f in common_footer:
                            lines = lines[:-1]
                    raw = "\n".join(lines)
                    clean = clean_text(raw)
                    page_texts.append((i, clean))
                    all_text += " " + clean

        return page_texts, all_text.strip(), None

    except Exception as e:
        return [], "", f"Error membaca PDF: {e}"


# =============================================================================
# EKSTRAKSI BAB (CHAPTER EXTRACTION)
# =============================================================================

def extract_chapters(pdf_path, max_pages_for_toc=15, target_chunks=8):
    """
    Deteksi bab menggunakan TOC → heading → fallback adaptive.
    Return: list of (title, text_chunk)
    """
    def clean_line(line):
        return re.sub(r'\s+', ' ', line.strip())

    def is_heading(line):
        if len(line.split()) <= 10:
            if line.isupper():
                return True
            if re.match(r'^(BAB|Pasal|Bagian|Subbab)\s+[IVXLC\d]+', line):
                return True
            if re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$', line):
                return True
        return False

    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")

    total_pages = len(pages_text)

    # 1. TOC
    toc_entries = []
    for i in range(min(max_pages_for_toc, total_pages)):
        lines = [clean_line(l) for l in pages_text[i].split("\n") if l.strip()]
        for line in lines:
            m = re.match(r'^(BAB\s+[IVXLC\d]+.*?)(?:\.{2,}|\s+)(\d+)', line)
            if m:
                toc_entries.append((m.group(1), int(m.group(2))))

    if toc_entries:
        chapters = []
        for idx, (title, start) in enumerate(toc_entries):
            end = toc_entries[idx + 1][1] if idx + 1 < len(toc_entries) else total_pages
            text = "\n".join(pages_text[start - 1: end])
            chapters.append((title, text))
        print(f"✅ Menggunakan TOC ({len(chapters)} bab)")
        return chapters

    # 2. Heading Detection
    headings = []
    for i, text in enumerate(pages_text, start=1):
        lines = [clean_line(l) for l in text.split("\n") if l.strip()]
        for line in lines:
            if is_heading(line):
                headings.append((line, i))

    if headings:
        chapters = []
        for idx, (title, start) in enumerate(headings):
            end = headings[idx + 1][1] if idx + 1 < len(headings) else total_pages
            text = "\n".join(pages_text[start - 1: end])
            chapters.append((title, text))
        print(f"✅ Menggunakan deteksi heading ({len(chapters)} bab)")
        return chapters

    # 3. Fallback Adaptive
    pages_per_chunk = max(10, math.ceil(total_pages / target_chunks))
    chunks = []
    for i in range(0, total_pages, pages_per_chunk):
        start = i + 1
        end = min(i + pages_per_chunk, total_pages)
        text = "\n".join(pages_text[i:end])
        chunks.append((f"Halaman {start}-{end}", text))
    print(f"⚙️ Fallback adaptive: {len(chunks)} chunk")
    return chunks


# =============================================================================
# SUMMARIZATION (LexRank)
# =============================================================================

def summarize_lexrank(text, sentence_count=5):
    """Ekstraktif summarizer menggunakan LexRank."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(s) for s in summary)


def build_final_combined_summary(book_title, chapters, sentence_count=5):
    """
    Buat ringkasan per bab, lalu gabungkan.
    book_title dimasukkan di awal final_combined_summary.

    Format output:
        "<book_title>. <Bab 1>. <summary 1> <Bab 2>. <summary 2> ..."

    Args:
        book_title: Judul buku (string)
        chapters: list of (title, text_chunk) setelah clean_rumus
        sentence_count: Jumlah kalimat per bab untuk LexRank

    Returns:
        final_combined_summary (string), chapter_summaries (list of string)
    """
    chapter_summaries = []
    combined_parts = []

    for title, content in chapters:
        cleaned = clean_rumus(content)
        summary = summarize_lexrank(cleaned, sentence_count=sentence_count).strip() + "."
        chapter_summaries.append(summary)
        combined_text = f"{title}. {summary}"
        combined_parts.append(combined_text)

    # Sisipkan book_title di depan
    final_combined_summary = f"{book_title}. " + " ".join(combined_parts)

    print(f"\n✅ All summaries combined!")
    print(f"📊 Final length: {len(final_combined_summary):,} chars")
    print(f"📊 Approximately {len(final_combined_summary.split())} words")

    return final_combined_summary, chapter_summaries


# =============================================================================
# EKSTRAKSI KEYWORD KONTEKS DARI SUMMARY
# =============================================================================

def clean_book_context(text):
    """Bersihkan teks ringkasan sebelum diekstrak keywordnya."""
    text = text.lower()
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-z\s\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_summary_keyword(text, combined_stopwords_set, top_per_n=350):
    """
    Ekstrak keyword dari teks ringkasan (final_combined_summary).
    Filter: stopword Sastrawi+NLTK+STOPWORDS_ID, duplikat kata.
    Return: list of string (keyword bersih)
    """
    keywords_all = []
    for n in [1, 2, 3]:
        kw_extractor = yake.KeywordExtractor(lan="multilingual", n=n, top=top_per_n)
        keywords_all.extend(kw_extractor.extract_keywords(text))

    seen = set()
    cleaned = []
    for kw, score in sorted(keywords_all, key=lambda x: x[1]):
        kw = kw.strip()
        if not kw:
            continue
        words = kw.split()
        if len(words) > 1 and len(set(words)) == 1:
            continue
        if any(w.lower() in combined_stopwords_set for w in words):
            continue
        if any(w.lower() in STOPWORDS_ID for w in words):
            continue
        norm = kw.lower()
        if norm not in seen:
            seen.add(norm)
            cleaned.append((kw, score))
        cleaned = sorted(cleaned, key=lambda x: x[1])

    context_keywords_clean = [kw for kw, _ in cleaned]
    print(f"✓ Context keywords dari summary: {len(context_keywords_clean)}")
    return context_keywords_clean


# =============================================================================
# YAKE PIPELINE UNTUK TEKS BUKU
# =============================================================================

def filter_multiword_capitalized_phrases(keywords):
    """Pertahankan frasa multi-kata berkapital; kata tunggal semua lewat."""
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
    Hapus frasa multi-kata yang mengandung akronim (di posisi manapun)
    jika akronim tersebut sudah ada sebagai keyword tersendiri.
    """
    single_upper = {
        kw.upper() for kw, _ in keywords
        if len(kw.split()) == 1 and kw.isupper()
    }
    filtered = []
    for kw, score in keywords:
        words = kw.split()
        if len(words) > 1:
            if all(w.isupper() for w in words):
                filtered.append((kw, score))
                continue
            upper_words = [w.upper() for w in words]
            if any(u in single_upper for u in upper_words):
                continue
        filtered.append((kw, score))
    return filtered


def normalize_repeated_words_second(phrases):
    """Hilangkan kata berulang dalam frasa."""
    cleaned = []
    for ph in phrases:
        words = ph.split()
        no_consec = []
        for w in words:
            if not no_consec or w != no_consec[-1]:
                no_consec.append(w)
        unique = []
        seen = set()
        for w in no_consec:
            if w.lower() not in seen:
                seen.add(w.lower())
                unique.append(w)
        cleaned.append(" ".join(unique))
    return cleaned


def merge_reversed_phrases_second(phrases):
    """Gabung frasa dengan set kata sama (canonical form)."""
    normalized = {}
    for ph in phrases:
        key = tuple(sorted(ph.split()))
        if key not in normalized:
            normalized[key] = ph
    return list(normalized.values())


def is_base_word(word):
    """True jika kata adalah kata dasar (hasil stem == kata asli)."""
    w = word.lower().strip()
    if len(w) <= 3:
        return True
    return indonesian_stemmer.stem(w) == w


def filter_single_word_derivatives(keyphrases):
    """
    Filter keyword 1 kata: hanya pertahankan kata dasar.
    Kata multi-kata selalu dipertahankan.
    """
    return [
        ph for ph in keyphrases
        if len(ph.split()) > 1 or is_base_word(ph.split()[0])
    ]


def run_yake_pipeline(all_text, combined_stopwords_set, top_per_n=450):
    """
    Pipeline YAKE untuk teks buku (sesuai ipynb):
    raw YAKE → filter stopword → filter_multiword_capitalized
    → filter_phrases_with_acronyms → lowercase
    → normalize_repeated_words → merge_reversed_phrases
    → filter_single_word_derivatives
    """
    print("\n" + "=" * 70)
    print("YAKE PIPELINE")
    print("=" * 70)

    keywords_all = []
    for n in [1, 2, 3]:
        kw_extractor = yake.KeywordExtractor(lan="multilingual", n=n, top=top_per_n)
        keywords_all.extend(kw_extractor.extract_keywords(all_text))

    seen = set()
    cleaned = []
    for kw, score in sorted(keywords_all, key=lambda x: x[1]):
        kw = kw.strip()
        if not kw:
            continue
        words = kw.split()
        if len(words) > 1 and len(set(words)) == 1:
            continue
        if any(w.lower() in combined_stopwords_set for w in words):
            continue
        norm = kw.lower()
        if norm not in seen:
            seen.add(norm)
            cleaned.append((kw, score))
    print(f"✓ Raw YAKE: {len(cleaned)}")

    cleaned = filter_multiword_capitalized_phrases(cleaned)
    print(f"✓ After capitalize filter: {len(cleaned)}")

    cleaned = filter_phrases_with_existing_acronyms(cleaned)
    print(f"✓ After acronym filter: {len(cleaned)}")

    keyphrases = [kw.lower() for kw, _ in cleaned]
    keyphrases = normalize_repeated_words_second(keyphrases)
    keyphrases = merge_reversed_phrases_second(keyphrases)
    print(f"✓ After normalize & merge: {len(keyphrases)}")

    keyphrases = filter_single_word_derivatives(keyphrases)
    print(f"✓ After derivative filter: {len(keyphrases)}")

    print("=" * 70 + "\n")
    return keyphrases, cleaned


# =============================================================================
# FASTTEXT — FRASA VS FRASA
# =============================================================================

def normalize_text(text):
    return re.sub(r'[^a-z0-9\s]', ' ', text.lower()).strip()


def phrase_embedding_fasttext(phrase, model):
    tokens = normalize_text(phrase).split()
    if not tokens:
        return None
    vecs = [model.get_word_vector(t) for t in tokens]
    return np.mean(vecs, axis=0)


def compare_keywords_fasttext(keyphrases, context_keywords_clean, ft_model, threshold=0.5):
    """
    Bandingkan keyword buku (dari YAKE) dengan keyword konteks (dari summary)
    secara frasa-vs-frasa menggunakan FastText.

    Return: list of dict {'keyword_full', 'best_context_match', 'score', 'method'}
    """
    if ft_model is None:
        return [
            {'keyword_full': kw, 'best_context_match': None, 'score': 1.0, 'method': 'no-filter'}
            for kw in keyphrases
        ]

    # Precompute embedding konteks
    ctx_data = []
    for kw_ctx in context_keywords_clean:
        ctx_text = normalize_text(kw_ctx)
        if not ctx_text:
            continue
        vec = phrase_embedding_fasttext(ctx_text, ft_model)
        if vec is not None:
            ctx_data.append((ctx_text, vec))

    if not ctx_data:
        return [
            {'keyword_full': kw, 'best_context_match': None, 'score': 1.0, 'method': 'no-filter'}
            for kw in keyphrases
        ]

    matched = []
    for kw_full in keyphrases:
        full_text = normalize_text(kw_full)
        if not full_text:
            continue
        vec_full = phrase_embedding_fasttext(full_text, ft_model)
        if vec_full is None:
            continue

        best_sim = -1
        best_match = None
        for ctx_text, vec_ctx in ctx_data:
            sim = 1.0 if full_text == ctx_text else 1 - cosine(vec_full, vec_ctx)
            if sim > best_sim:
                best_sim = sim
                best_match = ctx_text

        if best_sim >= threshold:
            matched.append({
                'keyword_full': full_text,
                'best_context_match': best_match,
                'score': round(float(best_sim), 4),
                'method': 'fasttext-context'
            })

    return sorted(matched, key=lambda x: x['score'], reverse=True)


# =============================================================================
# PEMETAAN HALAMAN
# =============================================================================

def tokenize3(text):
    return set(re.findall(r"\w+", text.lower()))


def map_keywords_to_pages(matched_keywords, page_texts, overlap_threshold=0.8):
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
# EVALUASI
# =============================================================================

def preprocess_phrases(phrases):
    roman = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}
    cleaned = set()
    for ph in phrases:
        ph = re.sub(r"\(.*?\)", "", ph)
        ph = re.sub(r"[^\w\s]", " ", ph)
        ph = re.sub(r"\s+", " ", ph).lower().strip()
        if any(c.isalpha() for c in ph) and len(ph) > 1 and ph not in roman and not re.fullmatch(r"[a-zA-Z]", ph):
            cleaned.add(ph)
    return cleaned


def extract_keywords_from_index_file(gt_path):
    """Ekstrak keyword dari PDF indeks ground truth."""
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


def fuzzy_match_evaluation(generated, ground_truth, threshold=85):
    tp, fp, matched_gt = set(), set(), set()
    for gen in generated:
        best_match, best_score = None, 0
        for gt in ground_truth:
            if gt in matched_gt:
                continue
            t = 60 if len(gen) <= 3 else (75 if len(gen) <= 4 else threshold)
            score = fuzz.ratio(gen.lower(), gt.lower())
            if score >= t and score > best_score:
                best_score = score
                best_match = gt
        if best_match:
            tp.add(gen)
            matched_gt.add(best_match)
        else:
            fp.add(gen)
    fn = ground_truth - matched_gt
    return tp, fp, fn


def fasttext_embed(text, model):
    tokens = text.lower().split()
    if not tokens or model is None:
        return np.zeros(300)
    return np.mean([model.get_word_vector(t) for t in tokens], axis=0)


def evaluasi_indeks(gt_path, matched_keywords, ft_model=None, use_fuzzy=True):
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

    generated = preprocess_phrases([item['keyword_full'] for item in matched_keywords])
    ground_truth = extract_keywords_from_index_file(gt_path)

    if use_fuzzy:
        tp, fp, fn = fuzzy_match_evaluation(generated, ground_truth)
    else:
        tp = generated & ground_truth
        fp = generated - ground_truth
        fn = ground_truth - generated

    precision = len(tp) / len(generated) if generated else 0
    recall = len(tp) / len(ground_truth) if ground_truth else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    ft_sim = None
    if ft_model and generated and ground_truth:
        emb1 = np.array([fasttext_embed(p, ft_model) for p in generated])
        emb2 = np.array([fasttext_embed(p, ft_model) for p in ground_truth])
        sims = sk_cosine(emb1, emb2)
        ft_sim = round(float(sims.max(axis=1).mean()), 4)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "similarity": ft_sim,
        "true_positives": sorted(tp),
        "false_positives": sorted(fp),
        "false_negatives": sorted(fn),
        "fuzzy_matching_used": use_fuzzy,
    }


# =============================================================================
# PEMBUATAN PDF INDEKS
# =============================================================================

def create_index_pdf(keyword_pages, pdf_path, book_title):
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'Title', parent=styles['Heading1'],
        fontSize=14, textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=20, alignment=TA_CENTER
    )
    story = [
        Paragraph(f"Indeks Buku: {book_title}", title_style),
        Spacer(1, 12)
    ]
    data = [["Kata Kunci", "Halaman"]]
    for kw in sorted(keyword_pages.keys()):
        data.append([kw, ", ".join(map(str, keyword_pages[kw]))])

    table = Table(data, colWidths=[300, 200])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ]))
    story.append(table)
    doc.build(story)
    print(f"PDF indeks dibuat: {pdf_path}")


# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template(
        'index.html',
        keyword_pages=session.get('keyword_pages'),
        book_title=session.get('book_title'),
        download_file=session.get('download_file'),
        eval_results=session.get('eval_results'),
    )


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Menerima: file PDF buku + judul buku.
    Alur: preprocess → extract chapters → summarize (LexRank) →
          build final_combined_summary (dengan book_title) →
          extract context keywords → YAKE pipeline →
          FastText filter (frasa vs frasa) → map halaman.
    """
    if 'file_buku' not in request.files:
        flash('Tidak ada file yang diunggah.', 'error')
        return redirect(url_for('index'))

    file = request.files['file_buku']
    if file.filename == '':
        flash('Tidak ada file yang dipilih.', 'error')
        return redirect(url_for('index'))

    book_title = request.form.get('book_title', '').strip()
    if not book_title:
        flash('Judul buku harus diisi.', 'error')
        return redirect(url_for('index'))

    if not (file and allowed_file(file.filename)):
        flash('Format file tidak valid. Hanya PDF.', 'error')
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    print(f"File disimpan: {filepath}")

    try:
        # 1. Preprocessing PDF
        print("Preprocessing PDF...")
        page_texts, all_text, error = preprocess_pdf(filepath)
        if error:
            flash(f'Error: {error}', 'error')
            return redirect(url_for('index'))
        print(f"Teks diekstrak: {len(all_text)} karakter")

        # 2. Ekstraksi bab
        print("Ekstraksi bab...")
        chapters = extract_chapters(filepath)
        print(f"Jumlah bab: {len(chapters)}")

        # 3. Summarization + build final_combined_summary
        #    book_title disisipkan di awal sesuai permintaan
        print("Summarization per bab...")
        final_combined_summary, chapter_summaries = build_final_combined_summary(
            book_title, chapters, sentence_count=5
        )
        print(f"final_combined_summary: {len(final_combined_summary)} chars")

        # 4. Ekstrak keyword konteks dari final_combined_summary
        print("Ekstrak keyword konteks dari summary...")
        context_clean = clean_book_context(final_combined_summary)
        context_keywords_clean = extract_summary_keyword(
            context_clean, combined_stopwords, top_per_n=250
        )

        # 5. YAKE pipeline pada teks buku
        print("YAKE pipeline teks buku...")
        keyphrases, _ = run_yake_pipeline(all_text, combined_stopwords, top_per_n=450)

        # 6. Filter FastText frasa-vs-frasa
        print("FastText filter...")
        matched_keywords = compare_keywords_fasttext(
            keyphrases, context_keywords_clean, ft_model, threshold=0.5
        )
        print(f"Matched: {len(matched_keywords)}")

        # 7. Pemetaan halaman
        print("Pemetaan halaman...")
        keyword_pages = map_keywords_to_pages(matched_keywords, page_texts)
        print(f"Keyword dengan halaman: {len(keyword_pages)}")

        # Simpan sesi
        session['keyword_pages'] = keyword_pages
        session['matched_keywords'] = matched_keywords
        session['book_title'] = book_title
        session['download_file'] = filename
        session['buku_path'] = filepath
        session['final_combined_summary'] = final_combined_summary  # simpan juga

        flash(f'✅ Berhasil! Ditemukan {len(keyword_pages)} kata kunci.', 'success')

    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f'Terjadi error: {e}', 'error')

    return redirect(url_for('index'))


@app.route('/evaluasi', methods=['POST'])
def evaluasi():
    """Evaluasi F1 Score terhadap PDF indeks ground truth."""
    matched_keywords = session.get('matched_keywords')
    if not matched_keywords:
        flash('Sesi tidak ditemukan. Harap unggah buku terlebih dahulu.', 'error')
        return redirect(url_for('index'))

    if 'file_gt' not in request.files or request.files['file_gt'].filename == '':
        flash('Tidak ada file ground truth yang diunggah.', 'error')
        return redirect(url_for('index'))

    file_gt = request.files['file_gt']
    if not allowed_file(file_gt.filename):
        flash('Format ground truth harus PDF.', 'error')
        return redirect(url_for('index'))

    gt_filename = f"gt_{secure_filename(file_gt.filename)}"
    gt_path = os.path.join(app.config['UPLOAD_FOLDER'], gt_filename)
    file_gt.save(gt_path)

    use_fuzzy = request.form.get('use_fuzzy', 'true').lower() == 'true'

    try:
        eval_results = evaluasi_indeks(gt_path, matched_keywords, ft_model, use_fuzzy=use_fuzzy)
        session['eval_results'] = eval_results
        flash(f"✅ Evaluasi selesai. F1: {eval_results['f1_score']}", 'success')
    except Exception as e:
        flash(f'Error evaluasi: {e}', 'error')

    return redirect(url_for('index'))


@app.route('/api/search_phrase', methods=['POST'])
def api_search_phrase():
    """Cari frasa di dalam PDF buku, kembalikan nomor halaman."""
    buku_path = session.get('buku_path')
    if not buku_path or not os.path.exists(buku_path):
        return jsonify({'status': 'error', 'message': 'Sesi buku tidak ditemukan.'}), 400

    phrase = (request.json or {}).get('phrase', '').strip()
    if not phrase:
        return jsonify({'status': 'error', 'message': 'Frasa tidak boleh kosong.'}), 400

    found_pages = []
    try:
        with pdfplumber.open(buku_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and phrase.lower() in text.lower():
                    found_pages.append(i)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

    if found_pages:
        return jsonify({'status': 'success', 'pages': found_pages})
    return jsonify({'status': 'not_found', 'message': 'Indeks tidak ada di buku ini.'})


@app.route('/api/bulk_delete', methods=['POST'])
def api_bulk_delete():
    """Hapus entri indeks dari sesi."""
    current_index = session.get('keyword_pages', {})
    phrases = (request.json or {}).get('phrases', [])
    deleted = sum(1 for p in phrases if current_index.pop(p, None) is not None)
    session['keyword_pages'] = current_index
    session.modified = True
    return jsonify({'status': 'success', 'deleted_count': deleted})


@app.route('/api/download_selected_pdf', methods=['POST'])
def api_download_selected_pdf():
    """Buat dan kirim PDF hanya berisi frasa yang dipilih."""
    selected = (request.json or {}).get('phrases', [])
    if not selected:
        return jsonify({'status': 'error', 'message': 'Tidak ada frasa dipilih.'}), 400

    full_index = session.get('keyword_pages', {})
    book_title = session.get('book_title', 'Indeks')
    filtered = {p: full_index[p] for p in selected if p in full_index}

    if not filtered:
        return jsonify({'status': 'error', 'message': 'Data tidak ditemukan di sesi.'}), 404

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"indeks_terpilih_{ts}.pdf"
    pdf_path = os.path.join(app.config['RESULT_FOLDER'], pdf_filename)
    create_index_pdf(filtered, pdf_path, f"{book_title} (Terpilih)")
    return send_file(pdf_path, as_attachment=True, download_name=pdf_filename)


@app.route('/download/<original_filename>')
def download(original_filename):
    """Unduh seluruh indeks sebagai PDF."""
    keyword_pages = session.get('keyword_pages')
    book_title = session.get('book_title')
    if not keyword_pages or not book_title:
        flash('Sesi tidak ditemukan.', 'error')
        return redirect(url_for('index'))

    pdf_filename = f"indeks_{secure_filename(original_filename)}"
    pdf_path = os.path.join(app.config['RESULT_FOLDER'], pdf_filename)
    try:
        create_index_pdf(keyword_pages, pdf_path, book_title)
        return send_file(pdf_path, as_attachment=True)
    except Exception as e:
        flash(f'Error membuat PDF: {e}', 'error')
        return redirect(url_for('index'))


@app.route('/clear')
def clear_session():
    session.clear()
    return redirect(url_for('index'))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    if not os.path.exists(FASTTEXT_PATH):
        print("-" * 60)
        print(f"PERINGATAN: Model FastText tidak ditemukan di:")
        print(f"  {FASTTEXT_PATH}")
        print("Download: https://fasttext.cc/docs/en/crawl-vectors.html")
        print("File yang dibutuhkan: cc.id.300.bin")
        print("-" * 60)
    app.run(debug=True)
