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
app.secret_key = 'kunci-rahasia-anda-yang-sangat-aman' # Ganti dengan kunci rahasia yang kuat

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

# --- Pemuatan Model & Stopwords Global (Dilakukan sekali saat start) ---
print("Memuat stopwords...")
factory = StopWordRemoverFactory()
sastrawi_stop = set(factory.get_stop_words())
nltk_stop_id = set(stopwords.words("indonesian"))
nltk_stop_en = set(stopwords.words("english"))

# Tambahkan stopwords kustom di sini jika perlu
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

print("Memuat model FastText (mungkin perlu beberapa saat)...")
# Ganti path ini sesuai dengan lokasi model Anda
# Download dari: https://fasttext.cc/docs/en/crawl-vectors.html
# Untuk bahasa Inggris: cc.en.300.bin
# Untuk bahasa Indonesia: cc.id.300.bin
model_path_fasttext = r'C:\SKRIPSI (code)\models\cc.en.300.'

ft_model = None
try:
    if os.path.exists(model_path_fasttext):
        print(f"Memuat model FastText dari {model_path_fasttext}...")
        ft_model = fasttext.load_model(model_path_fasttext)
        print(f"Model FastText berhasil dimuat. Dimensi: {ft_model.get_dimension()}")
    else:
        print(f"PERINGATAN: File model FastText ({model_path_fasttext}) tidak ditemukan.")
        print("Fungsi yang bergantung pada embedding (pemfilteran konteks, evaluasi) mungkin gagal.")
        print("Download model dari: https://fasttext.cc/docs/en/crawl-vectors.html")
except Exception as e:
    print(f"Gagal memuat model FastText: {e}")
    print("Silakan periksa file model Anda.")

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

        # Hitung frekuensi
        even_freq = Counter(header_even)
        odd_freq = Counter(header_odd)
        footer_freq = Counter(footer_all)

        n_even = len(header_even) if header_even else 1
        n_odd = len(header_odd) if header_odd else 1
        n_footer = len(footer_all) if footer_all else 1

        common_even = {h for h, c in even_freq.items() if (c/n_even > 0.3) and h}
        common_odd = {h for h, c in odd_freq.items() if (c/n_odd > 0.3) and h}
        common_footer = {f for f, c in footer_freq.items() if (c/n_footer > 0.3) and f}

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
                    
                    # Hapus stopwords dinamis
                    words = clean.split()
                    filtered = [w for w in words if w.lower() not in dynamic_stopwords]
                    clean_filtered = " ".join(filtered)
                    
                    page_texts.append((i, clean_filtered))
                    all_text += clean_filtered + " "

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
    Frasa multi-kata akan dihitung sebagai rata-rata vektor kata penyusunnya.
    
    Args:
        phrase: String frasa (bisa 1 kata atau lebih)
        ft_model: Model FastText yang sudah dimuat
    
    Returns:
        numpy array: Vektor embedding atau None jika gagal
    """
    if ft_model is None:
        return None
        
    tokens = normalize_text(phrase).split()
    if not tokens:
        return None
    
    # Ambil vektor untuk setiap token menggunakan get_word_vector
    # FastText akan handle OOV dengan subword information
    vecs = [ft_model.get_word_vector(token) for token in tokens]
    
    # Rata-rata vektor untuk multi-word phrase
    return np.mean(vecs, axis=0)

# =============================================================================
# FUNGSI EKSTRAKSI DAN EKSPANSI AKRONIM
# =============================================================================

def extract_acronyms(text):
    """
    Ekstrak pemetaan akronim dari teks.
    Contoh: "Weighted Sum Model (WSM)" -> {'wsm': 'weighted sum model'}
    
    Args:
        text: String teks yang mengandung akronim
    
    Returns:
        dict: Pemetaan akronim ke bentuk panjangnya
    """
    pattern = r'([A-Za-z][A-Za-z\s\-]+)\s*\(([A-Z]{2,})\)'
    matches = re.findall(pattern, text)

    acronym_map = {}
    for long_form, short in matches:
        acronym_map[short.lower()] = long_form.lower().strip()

    return acronym_map

def expand_acronym_if_needed(phrase, acronym_map):
    """
    Ekspansi akronim jika ditemukan dalam pemetaan.
    
    Args:
        phrase: String frasa yang mungkin akronim
        acronym_map: Dict pemetaan akronim
    
    Returns:
        String: Frasa yang sudah diexpand atau frasa asli
    """
    p = phrase.lower().strip()
    return acronym_map.get(p, p)

# =============================================================================
# FUNGSI PEMFILTERAN BERBASIS KONTEKS (FASTTEXT)
# =============================================================================

def compare_context_with_keywords_fasttext(
    book_title,
    book_summary,
    keywords,
    ft_model,
    threshold=0.3
):
    """
    Membandingkan kesamaan semantik konteks buku (judul + ringkasan)
    dengan keyword menggunakan FastText.
    
    Args:
        book_title: String judul buku
        book_summary: String ringkasan/abstrak buku
        keywords: List keyword hasil YAKE
        ft_model: Model FastText
        threshold: Threshold similarity minimum (default 0.3)
    
    Returns:
        List: Tuple (keyword, similarity_score, method)
    """
    if ft_model is None:
        print("PERINGATAN: Model FastText tidak tersedia. Mengembalikan semua keyword.")
        return [(kw, 1.0, "no-filter") for kw in keywords]
    
    # Gabungkan judul dan ringkasan sebagai konteks
    book_context = f"{book_title} {book_summary}"
    
    # Ekstrak akronim dari konteks
    acronym_map = extract_acronyms(book_context)
    
    # Buat embedding untuk konteks
    context_vec = phrase_embedding_fasttext(book_context, ft_model)
    if context_vec is None:
        print("PERINGATAN: Gagal membuat embedding konteks. Mengembalikan semua keyword.")
        return [(kw, 1.0, "no-filter") for kw in keywords]

    matched = []

    for kw in keywords:
        # Expand akronim jika ada
        kw_expanded = expand_acronym_if_needed(kw, acronym_map)
        
        # Buat embedding untuk keyword (sudah di-expand jika ada)
        kw_vec = phrase_embedding_fasttext(kw_expanded, ft_model)
        if kw_vec is None:
            continue

        # Hitung cosine similarity
        sim = 1 - cosine(context_vec, kw_vec)

        # Filter berdasarkan threshold
        if sim >= threshold:
            matched.append((kw, float(sim), "fasttext-context"))

    # Urutkan berdasarkan similarity (descending)
    return sorted(matched, key=lambda x: x[1], reverse=True)

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
        matched_keywords: List tuple (keyword, score, method)
        page_texts: List tuple (page_num, page_text)
        overlap_threshold: Threshold overlap minimum (default 0.8)
    
    Returns:
        dict: keyword -> list nomor halaman
    """
    matched_phrases = [kw for kw, _, _ in matched_keywords]
    keyword_pages = defaultdict(list)

    for phrase in matched_phrases:
        phrase_tokens = tokenize3(phrase)
        if not phrase_tokens:
            continue

        for page_num, page_txt in page_texts:
            page_tokens = tokenize3(page_txt)
            if not page_tokens:
                continue

            # Hitung overlap ratio
            overlap = len(phrase_tokens & page_tokens) / len(phrase_tokens)

            # Cocokkan berdasarkan threshold atau exact match
            if overlap >= overlap_threshold or phrase.lower() in page_txt.lower():
                if page_num not in keyword_pages[phrase]:
                    keyword_pages[phrase].append(page_num)

    # Urutkan halaman
    for phrase in keyword_pages:
        keyword_pages[phrase].sort()

    return dict(keyword_pages)

# =============================================================================
# FUNGSI YAKE PIPELINE (Ekstraksi dan Post-Processing)
# =============================================================================

def filter_stemmed_keywords(tfidf_results, stemmer_obj):
    """Filter keyword yang sudah di-stem."""
    filtered_results = {}
    for title, keywords in tfidf_results.items():
        filtered_kws = []
        for kw, score in keywords:
            stemmed = stemmer_obj.stem(kw.lower())
            if stemmed == kw.lower():
                filtered_kws.append((kw, score))
        filtered_results[title] = filtered_kws
    return filtered_results

def merge_reversed_phrases(keywords):
    """
    Gabungkan frasa yang memiliki set kata sama (canonical form deduplication).
    Contoh: 'hukum masyarakat' dan 'masyarakat hukum' → ambil yang skor terbaik
    """
    groups = {}
    for kw, score in keywords:
        if not kw or not kw.strip(): 
            continue
        words = tuple(w for w in kw.split() if w)
        key = tuple(sorted([w.lower() for w in words]))
        if key not in groups or score < groups[key][1]:
            groups[key] = (kw, score)
    merged = list(groups.values())
    merged.sort(key=lambda x: x[1])
    return merged

def normalize_repeated_words(keywords):
    """
    Token deduplication: Hilangkan kata berulang dalam frasa.
    Contoh: 'model fuzzy model' → 'model fuzzy'
    """
    cleaned = []
    for kw, score in keywords:
        if not kw or not kw.strip(): 
            continue
        words = [w for w in kw.split() if w]
        
        # Hapus duplikat berurutan
        no_consecutive = []
        for w in words:
            if not no_consecutive or w != no_consecutive[-1]:
                no_consecutive.append(w)
        
        # Hapus duplikat tersebar
        seen = set()
        unique_order = []
        for w in no_consecutive:
            lower_w = w.lower()
            if lower_w not in seen:
                seen.add(lower_w)
                unique_order.append(w)
        
        cleaned_kw = " ".join(unique_order)
        cleaned.append((cleaned_kw, score))
    return cleaned

def filter_multiword_capitalized_phrases(keywords):
    """
    Named entity filtering: Pertahankan frasa multi-kata yang ada kapitalisasi.
    Contoh: 'Fuzzy Logic System' → pertahankan, 'fuzzy logic system' → hapus
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

def filter_by_capitalization(keywords):
    """
    Case-insensitive deduplication dengan preferensi kapitalisasi.
    """
    grouped = {}
    for kw, score in keywords:
        key_lower = kw.lower()
        if key_lower not in grouped:
            grouped[key_lower] = (kw, score)
        else:
            _, existing_score = grouped[key_lower]
            if score < existing_score:
                grouped[key_lower] = (kw, score)
    
    final_keywords = []
    for key_lower, (kw, score) in grouped.items():
        if any(k.upper() == k and k.lower() == key_lower for k, _ in keywords):
            if kw.upper() == kw:
                final_keywords.append((kw, score))
        else:
            final_keywords.append((kw, score))
    return final_keywords

def filter_phrases_with_existing_acronyms(keywords):
    """
    Redundancy removal: Hapus frasa yang mengandung akronim jika akronim sudah ada.
    Contoh: Jika 'AHP' ada, hapus 'metode AHP'
    """
    single_upper_keywords = {
        kw.upper() for kw, _ in keywords
        if len(kw.split()) == 1 and kw.isupper()
    }
    
    filtered = []
    for kw, score in keywords:
        words = kw.split()
        if len(words) > 1:
            all_upper = all(w.isupper() for w in words)
            if all_upper:
                filtered.append((kw, score))
                continue
            
            upper_words = [w.upper() for w in words]
            should_remove = any(
                u in single_upper_keywords and i != 0
                for i, u in enumerate(upper_words)
            )
            if should_remove: 
                continue
        
        filtered.append((kw, score))
    return filtered

def extract_initials_from_title_phrase(phrase):
    """
    Ekstrak inisial dari kata yang diawali huruf kapital.
    Contoh: 'Analytic Hierarchy Process' → 'AHP'
    """
    tokens = re.findall(r"\w+", phrase)
    initials = []
    for t in tokens:
        if t and (t[0].isupper() or t.isupper()):
            initials.append(t[0].upper())
    return "".join(initials)

def normalize_acronym(a):
    """Acronym normalization: A.H.P → AHP"""
    return re.sub(r"\.", "", a).upper()

def boost_full_phrases_from_acronyms_v2(matched_keywords, boost_factor=0.7):
    """
    Feature boosting: Boost frasa panjang yang merupakan kepanjangan dari akronim.
    Contoh: Jika 'AHP' ada, boost 'Analytic Hierarchy Process'
    """
    # Kumpulkan akronim
    acronyms = set()
    for kw, _ in matched_keywords:
        kw_stripped = kw.strip()
        if re.fullmatch(r"(?:[A-Z]+\.)*[A-Z]+", kw_stripped):
            acronyms.add(normalize_acronym(kw_stripped))
    
    # Boost frasa yang cocok dengan akronim
    boosted = []
    for kw, score in matched_keywords:
        new_score = score
        if len(kw.split()) > 1:
            initials = extract_initials_from_title_phrase(kw)
            norm_initials = normalize_acronym(initials)
            if norm_initials and norm_initials in acronyms:
                new_score = score * boost_factor
        boosted.append((kw, new_score))
    return boosted

def filter_similar_phrases_by_overlap_safe(matched_keywords, min_common=2, min_overlap_ratio=0.6):
    """
    Fuzzy deduplication: Hapus frasa yang terlalu mirip berdasarkan overlap kata.
    """
    matched_keywords_lower = [(kw.lower(), score, kw) for kw, score in matched_keywords]
    matched_keywords_lower = sorted(matched_keywords_lower, key=lambda x: x[1])
    
    neutral_words = {
       "system", "sistem", "decision", "support", "sum", "product", 
       "process", "fuzzy", "mabac", "ahp", "edas", "wsm", "multi-criteria", "making"
    }
    
    to_remove_indices = set()
    
    for i, (kw1_lower, score1, _) in enumerate(matched_keywords_lower):
        if i in to_remove_indices: 
            continue
        words1 = set(kw1_lower.split())
        
        for j, (kw2_lower, score2, _) in enumerate(matched_keywords_lower[i + 1:], start=i + 1):
            if j in to_remove_indices: 
                continue
            words2 = set(kw2_lower.split())
            common = words1 & words2
            
            # Skip jika hanya overlap kata netral
            if common and all(w in neutral_words for w in common): 
                continue
            
            overlap_ratio = 0.0
            if min(len(words1), len(words2)) > 0:
                 overlap_ratio = len(common) / min(len(words1), len(words2))
            
            if len(common) >= min_common and overlap_ratio >= min_overlap_ratio and common != {"weighted", "model"}:
                if len(words2) > len(words1) or score2 < score1:
                    to_remove_indices.add(i)
                else:
                    to_remove_indices.add(j)
    
    filtered = [
        (orig_kw, s) for idx, (_, s, orig_kw) in enumerate(matched_keywords_lower)
        if idx not in to_remove_indices
    ]
    return sorted(filtered, key=lambda x: x[1])

def build_page_map(valid_phrases, page_texts, threshold=0.8):
    """
    Mapping frasa ke halaman kemunculannya.
    """
    page_map = defaultdict(list)
    for phrase in valid_phrases:
        phrase_tokens = tokenize2(phrase)
        if not phrase_tokens: 
            continue
        for page_num, page_txt in page_texts:
            page_tokens = tokenize2(page_txt)
            overlap = len(phrase_tokens & page_tokens) / len(phrase_tokens)
            if overlap >= threshold:
                page_map[phrase].append(page_num)
    return page_map

def merge_by_common_sequence(words1, words2):
    """
    Helper untuk merge_related_phrases: Gabung frasa dengan urutan kata yang sama.
    """
    best_merge = None
    max_common_len = 0
    
    for i in range(len(words1)):
        for j in range(len(words2)):
            k = 0
            while i + k < len(words1) and j + k < len(words2) and words1[i + k] == words2[j + k]:
                k += 1
            
            if k >= 2 and k > max_common_len:
                max_common_len = k
                merged = None
                if i == 0 and j > 0:
                    merged = words2[:j] + words1
                elif j == 0 and i > 0:
                    merged = words1[:i] + words2
                elif j > 0 and i > 0:
                    continue
                else:
                    merged = words1[:i] + words2[j:]
                best_merge = merged
    
    return best_merge

def merge_related_phrases(matched_keywords, page_map, page_texts, all_text_lower, overlap_ratio=0.6):
    """
    Phrase consolidation: Gabung frasa yang saling berhubungan.
    Contoh: 'decision support' + 'support system' → 'decision support system'
    """
    def find_phrase_pages(phrase):
        found = []
        for page_num, text in page_texts:
            if phrase.lower() in text.lower():
                found.append(page_num)
        return found
    
    merged = []
    used = set()
    original_case_map = {kw.lower(): kw for kw, _ in matched_keywords}
    
    for i, (kw1_orig, score1) in enumerate(matched_keywords):
        kw1 = kw1_orig.lower()
        if kw1 in used: 
            continue
        words1 = kw1.split()
        if len(words1) < 3: 
            continue
        
        for j, (kw2_orig, score2) in enumerate(matched_keywords[i+1:], start=i+1):
            kw2 = kw2_orig.lower()
            if kw2 in used: 
                continue
            words2 = kw2.split()
            if len(words2) < 3: 
                continue
            
            common = set(words1) & set(words2)
            if len(common) < 2: 
                continue
            
            merged_words = merge_by_common_sequence(words1, words2)
            if not merged_words: 
                continue
            
            merged_phrase_lower = " ".join(merged_words)
            if merged_phrase_lower not in all_text_lower: 
                continue
            
            pages1 = set(page_map.get(kw1_orig, []))
            pages2 = set(page_map.get(kw2_orig, []))
            pages3 = set(page_map.get(merged_phrase_lower, []))
            
            if not pages3:
                 pages3 = set(page_map.get(original_case_map.get(merged_phrase_lower, merged_phrase_lower), []))
            if not pages3:
                pages3 = set(find_phrase_pages(merged_phrase_lower))
            
            union_pages = pages1 | pages2
            if not union_pages: 
                continue
            
            ratio = len(pages3 & union_pages) / len(union_pages)
            
            if ratio >= overlap_ratio:
                merged_phrase_orig = original_case_map.get(merged_phrase_lower, merged_phrase_lower.title())
                merged.append((merged_phrase_orig, min(score1, score2)))
                used.add(kw1)
                used.add(kw2)
    
    # Tambahkan frasa yang tidak di-merge
    final_list = merged[:]
    merged_kw_lower = {kw.lower() for kw, _ in merged}
    for kw_orig, score in matched_keywords:
        kw_lower = kw_orig.lower()
        if kw_lower not in merged_kw_lower and kw_lower not in used:
            final_list.append((kw_orig, score))
    
    return final_list

def merge_reversed_phrases_second(phrases):
    """Versi kedua dari merge_reversed_phrases untuk normalisasi akhir."""
    normalized = {}
    for ph in phrases:
        words = ph.split()
        key = tuple(sorted(words))
        if key not in normalized:
            normalized[key] = ph
    return list(normalized.values())

def normalize_repeated_words_second(phrases):
    """Versi kedua dari normalize_repeated_words untuk normalisasi akhir."""
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

def run_yake_pipeline(all_text, page_texts, combined_stopwords_set, top_per_n=370):
    """
    Pipeline lengkap ekstraksi kata kunci menggunakan YAKE dengan post-processing.
    
    Args:
        all_text: String teks lengkap buku
        page_texts: List tuple (page_num, page_text)
        combined_stopwords_set: Set stopwords
        top_per_n: Jumlah top keyword per n-gram
    
    Returns:
        tuple: (keyphrases_final_strings, keyphrases_tuples)
    """
    print("\n" + "="*80)
    print("MEMULAI YAKE PIPELINE")
    print("="*80)
    
    # 1. YAKE Extraction
    keywords_all = []
    for n in [1, 2, 3]:
        kw_extractor = yake.KeywordExtractor(lan="multilingual", n=n, top=top_per_n)
        keywords = kw_extractor.extract_keywords(all_text)
        keywords_all.extend(keywords)
    
    # 2. Initial Cleaning
    seen = set()
    cleaned_keywords = []
    for kw, score in sorted(keywords_all, key=lambda x: x[1]):
        kw = kw.strip()
        if not kw: 
            continue
        words = kw.split()
        
        # Filter frasa tidak bermakna
        if len(words) > 1 and len(set(words)) == 1: 
            continue
        if any(w.lower() in combined_stopwords_set for w in words): 
            continue
        
        norm_kw = kw.lower()
        if norm_kw not in seen:
            seen.add(norm_kw)
            cleaned_keywords.append((kw, score))
    
    cleaned_keywords = sorted(cleaned_keywords, key=lambda x: x[1])
    print(f"✓ YAKE (raw): {len(cleaned_keywords)} keywords")
    
    # 3. Normalization & Deduplication
    cleaned_keywords_2 = normalize_repeated_words(cleaned_keywords)
    cleaned_keywords_2 = merge_reversed_phrases(cleaned_keywords_2)
    print(f"✓ After norm/merge reverse: {len(cleaned_keywords_2)} keywords")
    
    # 4. Named Entity Filtering
    capital_phrase_filtered_keywords = filter_multiword_capitalized_phrases(cleaned_keywords_2)
    print(f"✓ After filter non-capitalized: {len(capital_phrase_filtered_keywords)} keywords")
    
    # 5. Capitalization Deduplication
    capital_filtered_keywords = filter_by_capitalization(capital_phrase_filtered_keywords)
    print(f"✓ After filter by capitalization: {len(capital_filtered_keywords)} keywords")
    
    # 6. Acronym Redundancy Removal
    capital_filtered_keywords_acronym = filter_phrases_with_existing_acronyms(capital_filtered_keywords)
    print(f"✓ After filter by acronym: {len(capital_filtered_keywords_acronym)} keywords")
    
    # 7. Feature Boosting
    capital_filtered_keywords_boosted = boost_full_phrases_from_acronyms_v2(
        capital_filtered_keywords_acronym, boost_factor=0.7
    )
    print(f"✓ After boost phrases: {len(capital_filtered_keywords_boosted)} keywords")
    
    # 8. Fuzzy Deduplication
    filtered_keywords = filter_similar_phrases_by_overlap_safe(capital_filtered_keywords_boosted, min_common=2)
    print(f"✓ After filter similar overlap: {len(filtered_keywords)} keywords")
    
    # 9. Page Mapping
    page_map_yake = build_page_map([kw for kw, _ in filtered_keywords], page_texts)
    
    # 10. Phrase Consolidation
    appended_keywords = merge_related_phrases(
        filtered_keywords, 
        page_map_yake, 
        page_texts, 
        all_text.lower()
    )
    print(f"✓ After merge related: {len(appended_keywords)} keywords")
    
    # 11. Final Normalization
    keyphrases_tuples = appended_keywords
    keyphrases_final_strings = [kw for kw, _ in keyphrases_tuples]
    keyphrases_final_strings = normalize_repeated_words_second(keyphrases_final_strings)
    keyphrases_final_strings = merge_reversed_phrases_second(keyphrases_final_strings)
    
    print(f"✓ Final YAKE phrases: {len(keyphrases_final_strings)} keywords")
    print("="*80 + "\n")
    
    return keyphrases_final_strings, keyphrases_tuples

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
    
    Args:
        set1: Set frasa pertama (hasil sistem)
        set2: Set frasa kedua (ground truth)
        ft_model: Model FastText
    
    Returns:
        float: Rata-rata similarity
    """
    if not set1 or not set2 or ft_model is None:
        return 0.0

    from sklearn.metrics.pairwise import cosine_similarity
    
    emb1 = np.array([fasttext_embed(p, ft_model) for p in set1])
    emb2 = np.array([fasttext_embed(p, ft_model) for p in set2])

    sims = cosine_similarity(emb1, emb2)

    # Ambil similarity terbaik untuk setiap keyword sistem
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
    Ekstrak keyword dari file indeks (PDF).
    Menggabungkan baris terputus dan membersihkan format.
    """
    doc = fitz.open(gt_path)
    keywords = set()

    for page in doc:
        text = page.get_text("text")
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        merged_lines = []
        buffer = ""

        for line in lines:
            # Skip heading huruf indeks
            if re.match(r"^[A-Z]$", line):
                continue

            if buffer:
                if not re.search(r",\s*\d", buffer) and not re.match(r"^[A-Z]\s*$", line):
                    buffer += " " + line
                    if re.search(r",\s*\d", line):
                        merged_lines.append(buffer)
                        buffer = ""
                    continue
                else:
                    merged_lines.append(buffer)
                    buffer = ""

            if not re.search(r",\s*\d", line) and not re.match(r"^[A-Z]$", line):
                buffer = line
            else:
                merged_lines.append(line)

        if buffer:
            merged_lines.append(buffer)

        # Ambil keyword
        for line in merged_lines:
            match = re.match(r"([A-Za-z\s\-\(\)]+)[,\s0-9ivxIVX]*$", line.strip())
            if match:
                kw = match.group(1).strip()
                if len(kw) > 1:
                    keywords.add(kw)

    return preprocess_phrases(keywords)

def fuzzy_match_evaluation(generated_keywords, ground_truth_keywords, threshold=85):
    """
    Evaluasi menggunakan fuzzy matching untuk mengatasi typo.
    
    Args:
        generated_keywords: Set keyword hasil sistem
        ground_truth_keywords: Set keyword ground truth
        threshold: Threshold similarity minimum (default 85)
    
    Returns:
        tuple: (tp_set, fp_set, fn_set) - set keyword yang match/tidak match
    """
    from rapidfuzz import fuzz
    
    tp = set()
    fp = set()
    matched_gt = set()
    
    # Untuk setiap keyword hasil sistem
    for gen_kw in generated_keywords:
        best_match = None
        best_score = 0
        
        # Cari ground truth yang paling mirip
        for gt_kw in ground_truth_keywords:
            if gt_kw in matched_gt:
                continue
                
            # Adaptive threshold untuk kata pendek
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
    
    # False negative: ground truth yang tidak ter-match
    fn = ground_truth_keywords - matched_gt
    
    return tp, fp, fn

def evaluasi_indeks(gt_path, matched_keywords, ft_model=None, use_fuzzy=True):
    """
    Evaluasi hasil indeks buku terhadap indeks ground truth.
    
    Args:
        gt_path: Path ke file ground truth (PDF)
        matched_keywords: List tuple (keyword, score, method)
        ft_model: Model FastText untuk similarity
        use_fuzzy: Boolean, gunakan fuzzy matching atau tidak
    
    Returns:
        dict: Hasil evaluasi (precision, recall, f1, similarity, dll)
    """
    # Ambil hasil sistem
    generated_keywords = preprocess_phrases([kw for kw, _, _ in matched_keywords])

    # Ambil ground truth
    ground_truth_keywords = extract_keywords_from_index_file(gt_path)

    # Evaluasi dengan atau tanpa fuzzy matching
    if use_fuzzy:
        print("Menggunakan fuzzy matching untuk evaluasi...")
        tp, fp, fn = fuzzy_match_evaluation(generated_keywords, ground_truth_keywords)
        tp_count = len(tp)
        fp_count = len(fp)
        fn_count = len(fn)
    else:
        print("Menggunakan exact matching untuk evaluasi...")
        tp = generated_keywords & ground_truth_keywords
        fp = generated_keywords - ground_truth_keywords
        fn = ground_truth_keywords - generated_keywords
        tp_count = len(tp)
        fp_count = len(fp)
        fn_count = len(fn)

    # Hitung metrik
    precision = tp_count / len(generated_keywords) if generated_keywords else 0
    recall = tp_count / len(ground_truth_keywords) if ground_truth_keywords else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # Hitung similarity menggunakan FastText
    ft_sim = None
    if ft_model is not None:
        ft_sim = average_fasttext_similarity(
            generated_keywords,
            ground_truth_keywords,
            ft_model
        )

    result = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "similarity": round(ft_sim, 4) if ft_sim is not None else None,
        "true_positives": sorted(tp),
        "false_positives": sorted(fp),
        "false_negatives": sorted(fn),
        "fuzzy_matching_used": use_fuzzy
    }

    return result

# =============================================================================
# FUNGSI PEMBUATAN PDF INDEKS
# =============================================================================

def create_index_pdf(keyword_pages, pdf_path, book_title):
    """
    Membuat file PDF indeks buku.
    
    Args:
        keyword_pages: dict keyword -> list halaman
        pdf_path: Path output PDF
        book_title: Judul buku
    """
    try:
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Style untuk judul
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        # Tambahkan judul
        story.append(Paragraph(f"Indeks Buku: {book_title}", title_style))
        story.append(Spacer(1, 12))
        
        # Buat tabel indeks
        data = [["Kata Kunci", "Halaman"]]
        
        for keyword in sorted(keyword_pages.keys()):
            pages = keyword_pages[keyword]
            pages_str = ", ".join(map(str, pages))
            data.append([keyword, pages_str])
        
        # Style tabel
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
    # Ambil data dari session
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
    # Validasi file
    if 'file_buku' not in request.files:
        flash('Tidak ada file yang diunggah', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file_buku']
    if file.filename == '':
        flash('Tidak ada file yang dipilih', 'error')
        return redirect(url_for('index'))
    
    # Validasi input judul dan ringkasan
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
    
    # Ambil stopwords tambahan (opsional)
    extra_stopwords_input = request.form.get('extra_stopwords', '').strip()
    extra_stopwords = set()
    if extra_stopwords_input:
        extra_stopwords = {w.strip().lower() for w in extra_stopwords_input.split(',') if w.strip()}
    
    # Gabungkan dengan stopwords global
    dynamic_stopwords = combined_stopwords.union(extra_stopwords)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"File disimpan di: {filepath}")
        print(f"Judul buku: {book_title}")
        print(f"Ringkasan buku: {book_summary[:100]}...")
        print(f"Stopwords tambahan: {extra_stopwords}")
        
        # Proses PDF
        try:
            # 1. Preprocessing PDF
            print("Memulai preprocessing PDF...")
            page_texts, all_text, error = preprocess_pdf(filepath, dynamic_stopwords)
            
            if error:
                flash(f'Error: {error}', 'error')
                return redirect(url_for('index'))
            
            print(f"Teks berhasil diekstrak: {len(all_text)} karakter")
            
            # 2. Ekstraksi kata kunci dengan YAKE
            print("Ekstraksi kata kunci dengan YAKE...")
            keyphrases, keyphrases_tuples = run_yake_pipeline(
                all_text, 
                page_texts, 
                dynamic_stopwords, 
                top_per_n=370
            )
            
            print(f"Keyword hasil YAKE: {len(keyphrases)}")
            
            # 3. Filter berdasarkan konteks menggunakan FastText
            print("Memfilter kata kunci berdasarkan konteks dengan FastText...")
            matched_keywords = compare_context_with_keywords_fasttext(
                book_title=book_title,
                book_summary=book_summary,
                keywords=keyphrases,
                ft_model=ft_model,
                threshold=0.3
            )
            
            print(f"Keyword yang match dengan konteks: {len(matched_keywords)}")
            
            # 4. Pemetaan halaman
            print("Memetakan keyword ke halaman...")
            keyword_pages = map_keywords_to_pages(matched_keywords, page_texts)
            
            print(f"Keyword dengan halaman: {len(keyword_pages)}")
            
            # 5. Simpan ke sesi
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
        print(f"Error saat mencari frasa: {e}")
        return jsonify({'status': 'error', 'message': f'Gagal membaca PDF: {e}'}), 500

    if found_pages:
        return jsonify({'status': 'success', 'pages': found_pages})
    else:
        return jsonify({'status': 'not_found', 'message': 'Indeks tidak ada di buku ini.'})

@app.route('/api/bulk_delete', methods=['POST'])
def api_bulk_delete():
    """
    Menghapus item dari sesi berdasarkan list frasa.
    """
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
        print(f"Error bulk delete: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/download_selected_pdf', methods=['POST'])
def api_download_selected_pdf():
    """
    Membuat dan mengirim PDF yang hanya berisi frasa yang dipilih user.
    """
    try:
        data = request.json
        selected_phrases = data.get('phrases', [])
        
        if not selected_phrases:
            return jsonify({'status': 'error', 'message': 'Tidak ada frasa yang dipilih.'}), 400

        full_index = session.get('keyword_pages', {})
        book_title = session.get('book_title', 'Indeks Terpilih')
        
        # Filter data: Hanya ambil yang dipilih user
        filtered_data = {}
        for phrase in selected_phrases:
            if phrase in full_index:
                filtered_data[phrase] = full_index[phrase]
        
        if not filtered_data:
            return jsonify({'status': 'error', 'message': 'Data terpilih tidak ditemukan di sesi.'}), 404

        # Setup nama file dan path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"indeks_terpilih_{timestamp}.pdf"
        pdf_path = os.path.join(app.config['RESULT_FOLDER'], pdf_filename)
        
        # Buat PDF
        create_index_pdf(filtered_data, pdf_path, f"{book_title} (Terpilih)")
        
        return send_file(pdf_path, as_attachment=True, download_name=pdf_filename)

    except Exception as e:
        print(f"Error download selected: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/evaluasi', methods=['POST'])
def evaluasi():
    """Route untuk Evaluasi F1 Score."""
    
    # Ambil data dari sesi
    matched_keywords = session.get('matched_keywords')
    if not matched_keywords:
        flash('Sesi tidak ditemukan. Harap unggah file buku terlebih dahulu.', 'error')
        return redirect(url_for('index'))
    
    # Validasi file upload
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
        
        # Jalankan Evaluasi
        print("Memulai evaluasi...")
        if not ft_model:
            flash('Model FastText tidak dimuat, evaluasi similarity mungkin tidak akurat.', 'warning')
        
        # Pilih apakah menggunakan fuzzy matching atau tidak
        use_fuzzy = request.form.get('use_fuzzy', 'true').lower() == 'true'
        
        eval_results = evaluasi_indeks(gt_path, matched_keywords, ft_model, use_fuzzy=use_fuzzy)
        print("Evaluasi selesai.")
        
        if eval_results.get('error'):
            flash(eval_results['error'], 'error')
        
        # Simpan hasil evaluasi di sesi
        session['eval_results'] = eval_results
    
    return redirect(url_for('index'))

@app.route('/download/<original_filename>')
def download(original_filename):
    """Route untuk mengunduh hasil indeks (PDF)."""
    # Ambil data dari sesi
    keyword_pages = session.get('keyword_pages')
    book_title = session.get('book_title')
    
    if not keyword_pages or not book_title:
        flash('Sesi hasil indeks tidak ditemukan. Harap proses ulang file buku.', 'error')
        return redirect(url_for('index'))

    # Tentukan nama dan path file PDF
    pdf_filename = f"indeks_{secure_filename(original_filename)}"
    pdf_path = os.path.join(app.config['RESULT_FOLDER'], pdf_filename)

    try:
        # Buat PDF
        print(f"Membuat PDF on-demand di: {pdf_path}")
        create_index_pdf(keyword_pages, pdf_path, book_title)
        
        # Kirim file
        if os.path.exists(pdf_path):
            return send_file(pdf_path, as_attachment=True)
        else:
            flash('Gagal membuat file PDF.', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        print(f"Error saat membuat PDF on-demand: {e}")
        flash(f'Terjadi error saat membuat PDF: {e}', 'error')
        return redirect(url_for('index'))

@app.route('/clear')
def clear_session():
    """Membersihkan sesi dan memulai ulang."""
    session.clear()
    return redirect(url_for('index'))

# --- Jalankan Aplikasi ---
if __name__ == '__main__':
    # Validasi keberadaan model
    if not os.path.exists(model_path_fasttext):
        print("-" * 70)
        print(f"PERINGATAN: File model FastText '{model_path_fasttext}' tidak ditemukan.")
        print(f"Harap unduh model dan letakkan di: {os.path.dirname(model_path_fasttext)}")
        print("Download dari: https://fasttext.cc/docs/en/crawl-vectors.html")
        print("Untuk bahasa Inggris: cc.en.300.bin")
        print("Untuk bahasa Indonesia: cc.id.300.bin")
        print("-" * 70)
    
    app.run(debug=True)