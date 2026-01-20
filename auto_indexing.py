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

print("Memuat model Word2Vec (mungkin perlu beberapa saat)...")
# Ganti path ini sesuai dengan lokasi model Anda
model_path_gensim = r'C:\SKRIPSI (code)\models\cc.id.300.model' 
# model_path_gensim = r'C:\SKRIPSI (code)\models\cc.en.300.model' 
model_path_vec = 'cc.id.300.vec' # Pastikan file .vec ada di folder yang sama

w2v_model = None
try:
    if os.path.exists(model_path_gensim):
        print(f"Memuat model gensim dari {model_path_gensim}...")
        w2v_model = KeyedVectors.load(model_path_gensim)
    elif os.path.exists(model_path_vec):
        print(f"Memuat model .vec dari {model_path_vec}...")
        w2v_model = KeyedVectors.load_word2vec_format(model_path_vec)
        print("Menyimpan model ke format gensim untuk pemuatan lebih cepat berikutnya...")
        w2v_model.save(model_path_gensim)
    else:
        print(f"PERINGATAN: File model Word2Vec ({model_path_gensim} atau {model_path_vec}) tidak ditemukan.")
        print("Fungsi yang bergantung pada embedding (evaluasi, perbandingan) mungkin gagal.")
except Exception as e:
    print(f"Gagal memuat model Word2Vec: {e}")
    print("Silakan periksa file model Anda.")

print("Model dan stopwords berhasil dimuat. Aplikasi siap.")

# --- Fungsi Bantuan ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- SEMUA FUNGSI PEMROSESAN (clean_text, preprocess_pdf, dll.) ---
# ... (Fungsi-fungsi dari file Anda sebelumnya diletakkan di sini) ...
# ... (Saya akan mempersingkatnya agar muat) ...

def clean_text(text):
  text = re.sub(r'\bhlm\.?\s*\d+', '', text)
  text = re.sub(r'\bCet\.?\s*\d+', '', text)
  text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
  text = re.sub(r"\b[0-9ivxlcdm]+\b", " ", text)
  text = re.sub(r"\([^)]*\)", " ", text)
  text = re.sub(r"http\S+|www\S+", " ", text)
  text = re.sub(r"\.{3,}\s*\d+", " ", text)
  text = re.sub(r"[^A-Za-z0-9\s\.\-\,]", " ", text)  # perhatikan: titik tidak dihapus
  text = re.sub(r"\s+", " ", text).strip()
  text = re.sub(r"\b(wiki|wikipedia|org)\b", " ", text)
  return text

def normalize_line(line):
    line = line.lower().strip()
    line = re.sub(r"\d+", "", line)  # hapus angka
    line = re.sub(r"\s+", " ", line) # hapus spasi berlebih
    return line

def preprocess_pdf(pdf_path, dynamic_stopwords):
    """
    Ekstrak teks dari PDF, bersihkan header/footer, dan kembalikan teks per halaman.
    Menggunakan `dynamic_stopwords` untuk membersihkan teks.
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

                        if i % 2 == 0:  # halaman genap
                            header_even.append(header_line)
                        else:           # halaman ganjil
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
                    
                    page_texts.append((i, clean))
                    all_text += " " + clean

        return page_texts, all_text, None
    
    except Exception as e:
        print(f"Error saat memproses PDF: {e}")
        return [], "", f"Gagal memproses PDF: {e}"


def extract_chapters(pdf_path, max_pages_for_toc=15, pages_per_chunk=40):
    def join_wrapped_lines(lines):
        merged = []
        current = ""
        for i, line in enumerate(lines):
            clean = line.strip()
            if re.match(r'^(BAB\s+[IVXLC\d]+)', clean, re.IGNORECASE):
                if current: merged.append(current.strip())
                current = clean
            else:
                if current: current += " " + clean
                else: current = clean
            if re.search(r'(\.{2,}\s*\d+|\s+\d+)$', clean):
                merged.append(current.strip())
                current = ""
        if current: merged.append(current.strip())
        return [re.sub(r'\s+', ' ', m) for m in merged]

    def clean_line(line):
        return re.sub(r'\s+', ' ', line.strip())

    def is_heading(line):
        if len(line.split()) <= 10:
            if line.isupper(): return True
            if re.match(r'^(BAB|Pasal|Bagian|Subbab)\s+[IVXLC\d]+', line): return True
        return False

    pages_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages_text.append(text)
    except Exception as e:
        print(f"Gagal membuka PDF untuk ekstraksi bab: {e}")
        return []
    toc_entries = []
    for i in range(min(max_pages_for_toc, len(pages_text))):
        lines = [clean_line(l) for l in pages_text[i].split("\n") if l.strip()]
        merged = join_wrapped_lines(lines)
        for line in merged:
            m = re.match(r'^(BAB\s+[IVXLC\d]+.*?)(?:\.{2,}|\s+)(\d+)(?!\d)', line)
            if m:
                title = m.group(1).strip()
                try:
                    page_num = int(m.group(2))
                    toc_entries.append((title, page_num))
                except ValueError:
                    continue
    if toc_entries:
        chapters = []
        for idx, (title, start_page) in enumerate(toc_entries):
            end_page_num = toc_entries[idx + 1][1] if idx + 1 < len(toc_entries) else len(pages_text)
            text_chunk = "\n".join(pages_text[start_page-1 : end_page_num-1])
            chapters.append((title, text_chunk))
        print(f"✅ Menggunakan TOC ({len(chapters)} bab terdeteksi)")
        return chapters
    headings = []
    for i, text in enumerate(pages_text, start=1):
        lines = [clean_line(l) for l in text.split("\n") if l.strip()]
        for line in lines:
            if is_heading(line):
                headings.append((line, i))
    if headings:
        chapters = []
        for idx, (title, start_page) in enumerate(headings):
            end_page = headings[idx + 1][1] if idx + 1 < len(headings) else len(pages_text)
            text_chunk = "\n".join(pages_text[start_page-1 : end_page-1])
            chapters.append((title, text_chunk))
        print(f"✅ Menggunakan deteksi subjudul otomatis ({len(chapters)} bab terdeteksi)")
        return chapters
    chunks = [
        "\n".join(pages_text[i:i+pages_per_chunk])
        for i in range(0, len(pages_text), pages_per_chunk)
    ]
    chapters = [(f"Halaman {i*pages_per_chunk+1}-{(i+1)*pages_per_chunk}", text)
                for i, text in enumerate(chunks)]
    print(f"⚙️ Fallback ke pembagian per {pages_per_chunk} halaman ({len(chapters)} segmen)")
    return chapters

def extract_keywords_tfidf_per_chapter(chapters, stop_words_set, top_n=15):
    stopwords = set(stop_words_set)
    def clean_text2(text):
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    titles = []
    cleaned_texts = []
    if not chapters:
        return {}
    for title, text in chapters:
        titles.append(title)
        combined_text = f"{title} {text}"
        cleaned = clean_text2(combined_text)
        cleaned_texts.append(cleaned)
    vectorizer = TfidfVectorizer(
        stop_words=list(stopwords),
        ngram_range=(1, 4),
        max_df=0.7,
        min_df=1,
        token_pattern=r"(?u)\b[a-zA-Z0-9\-]{2,}\b"
    )
    X = vectorizer.fit_transform(cleaned_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())
    results = {}
    meaningless = {"ii","iii","iv","v","vi","vii","viii","ix","x",
                   "j","k","e","f","g","x","z","c","v","p", "bab"}
    for idx, title in enumerate(titles):
        row = X[idx].toarray().flatten()
        top_indices = row.argsort()[::-1][:top_n]
        keywords = [(feature_names[i], float(row[i])) for i in top_indices if row[i] > 0]
        title_terms = [w.lower() for w in re.findall(r"[a-zA-Z]+", title)]
        for t in title_terms:
            if t not in stopwords:
                if t not in feature_names:
                    keywords.append((t, 1.0))
                elif all(t not in k for k, _ in keywords):
                    keywords.append((t, 1.0))
        keywords = sorted(keywords, key=lambda x: x[1], reverse=True)[:top_n]
        keywords = [(k, s) for k, s in keywords if k not in meaningless and len(k) > 2]
        results[title] = keywords
    return results

def filter_stemmed_keywords(tfidf_results, stemmer_obj):
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
    groups = {}
    for kw, score in keywords:
        if not kw or not kw.strip(): continue
        words = tuple(w for w in kw.split() if w)
        key = tuple(sorted([w.lower() for w in words]))
        if key not in groups or score < groups[key][1]:
            groups[key] = (kw, score)
    merged = list(groups.values())
    merged.sort(key=lambda x: x[1])
    return merged

def normalize_repeated_words(keywords):
    cleaned = []
    for kw, score in keywords:
        if not kw or not kw.strip(): continue
        words = [w for w in kw.split() if w]
        no_consecutive = []
        for w in words:
            if not no_consecutive or w != no_consecutive[-1]:
                no_consecutive.append(w)
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
            if should_remove: continue
        filtered.append((kw, score))
    return filtered

def extract_initials_from_title_phrase(phrase):
    tokens = re.findall(r"\w+", phrase)
    initials = []
    for t in tokens:
        if t and (t[0].isupper() or t.isupper()):
            initials.append(t[0].upper())
    return "".join(initials)

def normalize_acronym(a):
    return re.sub(r"\.", "", a).upper()

def boost_full_phrases_from_acronyms_v2(matched_keywords, boost_factor=0.7):
    acronyms = set()
    for kw, _ in matched_keywords:
        kw_stripped = kw.strip()
        if re.fullmatch(r"(?:[A-Z]+\.)*[A-Z]+", kw_stripped):
            acronyms.add(normalize_acronym(kw_stripped))
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
    matched_keywords_lower = [(kw.lower(), score, kw) for kw, score in matched_keywords] # simpan original case
    matched_keywords_lower = sorted(matched_keywords_lower, key=lambda x: x[1])
    neutral_words = {
       "system", "sistem", "decision", "support", "sum", "product", 
       "process", "fuzzy", "mabac", "ahp", "edas", "wsm", "multi-criteria", "making"
    }
    to_remove_indices = set()
    for i, (kw1_lower, score1, _) in enumerate(matched_keywords_lower):
        if i in to_remove_indices: continue
        words1 = set(kw1_lower.split())
        for j, (kw2_lower, score2, _) in enumerate(matched_keywords_lower[i + 1:], start=i + 1):
            if j in to_remove_indices: continue
            words2 = set(kw2_lower.split())
            common = words1 & words2
            if common and all(w in neutral_words for w in common): continue
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

def tokenize2(text):
    return set(re.findall(r"\w+", text.lower()))

def build_page_map(valid_phrases, page_texts, threshold=0.8):
    page_map = defaultdict(list)
    for phrase in valid_phrases:
        phrase_tokens = tokenize2(phrase)
        if not phrase_tokens: continue
        for page_num, page_txt in page_texts:
            page_tokens = tokenize2(page_txt)
            overlap = len(phrase_tokens & page_tokens) / len(phrase_tokens)
            if overlap >= threshold:
                page_map[phrase].append(page_num)
    return page_map

def merge_by_common_sequence(words1, words2):
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
        if kw1 in used: continue
        words1 = kw1.split()
        if len(words1) < 3: continue
        for j, (kw2_orig, score2) in enumerate(matched_keywords[i+1:], start=i+1):
            kw2 = kw2_orig.lower()
            if kw2 in used: continue
            words2 = kw2.split()
            if len(words2) < 3: continue
            common = set(words1) & set(words2)
            if len(common) < 2: continue
            merged_words = merge_by_common_sequence(words1, words2)
            if not merged_words: continue
            merged_phrase_lower = " ".join(merged_words)
            if merged_phrase_lower not in all_text_lower: continue
            pages1 = set(page_map.get(kw1_orig, []))
            pages2 = set(page_map.get(kw2_orig, []))
            pages3 = set(page_map.get(merged_phrase_lower, [])) # Coba lower
            if not pages3:
                 pages3 = set(page_map.get(original_case_map.get(merged_phrase_lower, merged_phrase_lower), [])) # Coba original case
            if not pages3:
                pages3 = set(find_phrase_pages(merged_phrase_lower))
            union_pages = pages1 | pages2
            if not union_pages: continue # Hindari pembagian nol
            ratio = len(pages3 & union_pages) / len(union_pages)
            if ratio >= overlap_ratio:
                merged_phrase_orig = original_case_map.get(merged_phrase_lower, merged_phrase_lower.title())
                merged.append((merged_phrase_orig, min(score1, score2)))
                used.add(kw1)
                used.add(kw2)
    final_list = merged[:]
    merged_kw_lower = {kw.lower() for kw, _ in merged}
    for kw_orig, score in matched_keywords:
        kw_lower = kw_orig.lower()
        if kw_lower not in merged_kw_lower and kw_lower not in used:
            final_list.append((kw_orig, score))
    return final_list

def merge_reversed_phrases_second(phrases):
    normalized = {}
    for ph in phrases:
        words = ph.split()
        key = tuple(sorted(words))
        if key not in normalized:
            normalized[key] = ph
    return list(normalized.values())

def normalize_repeated_words_second(phrases):
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
    keywords_all = []
    for n in [1, 2, 3]:
        kw_extractor = yake.KeywordExtractor(lan="multilingual", n=n, top=top_per_n)
        keywords = kw_extractor.extract_keywords(all_text)
        keywords_all.extend(keywords)
    seen = set()
    cleaned_keywords = []
    for kw, score in sorted(keywords_all, key=lambda x: x[1]):
        kw = kw.strip()
        if not kw: continue
        words = kw.split()
        if len(words) > 1 and len(set(words)) == 1: continue
        if any(w.lower() in combined_stopwords_set for w in words): continue
        norm_kw = kw.lower()
        if norm_kw not in seen:
            seen.add(norm_kw)
            cleaned_keywords.append((kw, score))
    cleaned_keywords = sorted(cleaned_keywords, key=lambda x: x[1])
    print(f"YAKE (raw): {len(cleaned_keywords)} keywords")
    cleaned_keywords_2 = normalize_repeated_words(cleaned_keywords)
    cleaned_keywords_2 = merge_reversed_phrases(cleaned_keywords_2)
    print(f"After norm/merge reverse: {len(cleaned_keywords_2)} keywords")
    capital_phrase_filtered_keywords = filter_multiword_capitalized_phrases(cleaned_keywords_2)
    print(f"After filter non-capitalized: {len(capital_phrase_filtered_keywords)} keywords")
    capital_filtered_keywords = filter_by_capitalization(capital_phrase_filtered_keywords)
    print(f"After filter by capitalization: {len(capital_filtered_keywords)} keywords")
    capital_filtered_keywords_acronym = filter_phrases_with_existing_acronyms(capital_filtered_keywords)
    print(f"After filter by acronym: {len(capital_filtered_keywords_acronym)} keywords")
    capital_filtered_keywords_boosted = boost_full_phrases_from_acronyms_v2(
        capital_filtered_keywords_acronym, boost_factor=0.7
    )
    print(f"After boost phrases: {len(capital_filtered_keywords_boosted)} keywords")
    filtered_keywords = filter_similar_phrases_by_overlap_safe(capital_filtered_keywords_boosted, min_common=2)
    print(f"After filter similar overlap: {len(filtered_keywords)} keywords")
    page_map_yake = build_page_map([kw for kw, _ in filtered_keywords], page_texts)
    appended_keywords = merge_related_phrases(filtered_keywords, page_map_yake, page_texts, all_text.lower())
    print(f"After merge related: {len(appended_keywords)} keywords")
    keyphrases_tuples = appended_keywords
    keyphrases_final_strings = [kw for kw, _ in keyphrases_tuples]
    keyphrases_final_strings = normalize_repeated_words_second(keyphrases_final_strings)
    keyphrases_final_strings = merge_reversed_phrases_second(keyphrases_final_strings)
    print(f"Final YAKE phrases: {len(keyphrases_final_strings)} keywords")
    return keyphrases_final_strings, keyphrases_tuples

def domain_stem(word):
    ls = LancasterStemmer()
    custom_rules = {
        "fuzzification": "fuzzy", "defuzzification": "fuzzy", "fuzzifier": "fuzzy",
        "fuzzified": "fuzzy", "fuzziness": "fuzzy", "fuzzify": "fuzzy", "fuzzies": "fuzzy",
    }
    lw = word.lower()
    if lw in custom_rules: return custom_rules[lw]
    return ls.stem(lw)

def normalize_text(text):
    return re.sub(r'[^a-z0-9\s]', '', text.lower())

def normalize_phrase_order(phrase):
    words = phrase.split()
    return " ".join(sorted(words))

def normalize_case(phrase: str) -> str:
    return phrase.lower().strip()

def compare_title_with_keywords(title, tfidf_results, keywords, model, threshold=0.5):
    if not model:
        print("Model W2V tidak dimuat, perbandingan semantik dilewati.")
        matched = []
        title_norm = normalize_text(title)
        title_words = set(title_norm.split())
        for kw in keywords:
            kw_norm = normalize_text(kw)
            kw_tokens = set(kw_norm.split())
            if kw_tokens & title_words:
                matched.append((kw, 0.5, "string_fallback"))
        return sorted(matched, key=lambda x: x[1], reverse=True)
    title_norm = normalize_text(title)
    title_words = set(title_norm.split())
    normalized_phrases = [normalize_case(p) for p in keywords]
    tfidf_context = set()
    for _, kws in tfidf_results.items():
        for kw, score in kws:
            if score > 0.1:
                tfidf_context.update(normalize_text(kw).split())
    matched = []
    for kw in normalized_phrases:
        kw_norm = normalize_phrase_order(normalize_text(kw))
        kw_tokens = set(kw_norm.split())
        missing = [w for w in kw_norm.split() if w not in model]
        if missing:
            print(f"[!] Kata tidak ada di model W2V untuk '{kw}': {missing}")
        token_overlap = kw_tokens & tfidf_context
        ratio_tfidf = 0.0
        if token_overlap:
          ratio_tfidf = len(token_overlap) / len(kw_tokens)
        if ratio_tfidf >= 0.3:
          matched.append((kw, ratio_tfidf, "string+tfidf"))
          continue
        stemmed_tokens = set(domain_stem(w) for w in kw_norm.split())
        token_overlap_stem = stemmed_tokens & tfidf_context
        ratio_stem = len(token_overlap_stem) / len(stemmed_tokens) if stemmed_tokens else 0
        if ratio_stem >= 0.5:
            matched.append((kw, ratio_stem, "stemmed-tfidf"))
            continue
        try:
            kw_vec_tokens = [model[w] for w in kw_norm.split() if w in model]
            title_vec_tokens = [model[w] for w in title_words if w in model]
            context_vec_tokens = [model[w] for w in tfidf_context if w in model]
            if not kw_vec_tokens: continue
            kw_vec = np.mean(kw_vec_tokens, axis=0)
            sims = []
            if title_vec_tokens:
                title_vec = np.mean(title_vec_tokens, axis=0)
                sims.append(1 - cosine(kw_vec, title_vec))
            if context_vec_tokens:
                context_vec = np.mean(context_vec_tokens, axis=0)
                sims.append(1 - cosine(kw_vec, context_vec))
            if sims and max(sims) >= threshold:
                matched.append((kw, max(sims), "embedding+tfidf"))
        except Exception as e:
            print(f"Error W2V similarity: {e} untuk '{kw}'")
            pass
    return sorted(matched, key=lambda x: x[1], reverse=True)

# def filter_semantic_duplicates(matched_keywords, lang='id', similarity_threshold=0.9):
#     if not matched_keywords: return matched_keywords
#     keywords_map = {kw: (score, mtype) for kw, score, mtype in matched_keywords}
#     keywords = [kw for kw, _, _ in matched_keywords]
#     try:
#         translator = GoogleTranslator(source='auto', target=lang)
#         translated = [translator.translate(kw).lower().strip() for kw in keywords]
#     except Exception as e:
#         print(f"Gagal menerjemahkan: {e}. Menggunakan string asli.")
#         translated = [kw.lower().strip() for kw in keywords]
#     to_remove = set()
#     for i in range(len(keywords)):
#         if keywords[i] in to_remove: continue
#         for j in range(i + 1, len(keywords)):
#             if keywords[j] in to_remove: continue
#             sim = SequenceMatcher(None, translated[i], translated[j]).ratio()
#             if sim >= similarity_threshold:
#                 if len(keywords[i]) <= len(keywords[j]):
#                     to_remove.add(keywords[i])
#                 else:
#                     to_remove.add(keywords[j])
#     filtered = [(kw, score, mtype) for kw, score, mtype in matched_keywords if kw not in to_remove]
#     return filtered

def tokenize3(text):
    return set(re.findall(r"\w+", text.lower()))

def build_final_index(matched_keywords, page_texts):
    keyword_pages = defaultdict(list)
    phrase_map = {kw.lower(): kw for kw, _, _ in matched_keywords}
    tokenized_phrases = {}
    for phrase_lower in phrase_map.keys():
        tokenized_phrases[phrase_lower] = tokenize3(phrase_lower)
    for page_num, page_txt in page_texts:
        page_tokens = tokenize3(page_txt)
        if not page_tokens: continue
        for phrase_lower, phrase_tokens in tokenized_phrases.items():
            if not phrase_tokens: continue
            overlap = len(phrase_tokens & page_tokens) / len(phrase_tokens)
            if overlap >= 0.8 or phrase_lower in page_txt.lower():
                original_phrase = phrase_map[phrase_lower]
                if page_num not in keyword_pages[original_phrase]:
                    keyword_pages[original_phrase].append(page_num)
    for phrase in keyword_pages:
        keyword_pages[phrase].sort()
    return dict(keyword_pages)

def preprocess_phrases(phrases):
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

def hybrid_similarity(a, b, model):
    if not a or not b: return 0.0
    if not model:
        return fuzz.ratio(a.lower(), b.lower()) / 100.0
    a_tokens = [t for t in a.lower().split() if t in model]
    b_tokens = [t for t in b.lower().split() if t in model]
    if a_tokens and b_tokens:
        vec_a = np.mean([model[t] for t in a_tokens], axis=0)
        vec_b = np.mean([model[t] for t in b_tokens], axis=0)
        return float(1 - cosine(vec_a, vec_b))
    else:
        return fuzz.ratio(a.lower(), b.lower()) / 100.0

def extract_keywords_from_index_file(gt_path):
    try:
        doc = fitz.open(gt_path)
    except Exception as e:
        print(f"Gagal membuka file Ground Truth: {e}")
        return set()
    keywords = set()
    for page in doc:
        text = page.get_text("text")
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        merged_lines = []
        buffer = ""
        for line in lines:
            if re.match(r"^[A-Z]$", line): continue
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
        if buffer: merged_lines.append(buffer)
        for line in merged_lines:
            match = re.match(r"([A-Za-z\s\-\(\)]+)[,\s0-9ivxIVX]*$", line.strip())
            if match:
                kw = match.group(1).strip()
                if len(kw) > 1:
                    keywords.add(kw)
    return preprocess_phrases(keywords)

def average_hybrid_similarity(set1, set2, model):
    if not set1 or not set2: return 0.0
    sims = []
    for s1 in set1:
        best_sim = max(hybrid_similarity(s1, s2, model) for s2 in set2)
        sims.append(best_sim)
    return float(np.mean(sims)) if sims else 0.0

def evaluasi_indeks(gt_path, matched_keywords, model):
    generated_keywords = preprocess_phrases([kw for kw, _, _ in matched_keywords])
    ground_truth_keywords = extract_keywords_from_index_file(gt_path)
    if not ground_truth_keywords:
        return {
            "error": "Gagal membaca file ground truth atau file kosong.",
            "precision": 0, "recall": 0, "f1_score": 0, "hybrid_similarity": 0,
            "true_positives": [], "false_positives": sorted(generated_keywords), "false_negatives": []
        }
    tp = generated_keywords & ground_truth_keywords
    fp = generated_keywords - ground_truth_keywords
    fn = ground_truth_keywords - generated_keywords
    precision = len(tp) / len(generated_keywords) if generated_keywords else 0
    recall = len(tp) / len(ground_truth_keywords) if ground_truth_keywords else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    hybrid_sim = average_hybrid_similarity(generated_keywords, ground_truth_keywords, model)
    result = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "hybrid_similarity": round(hybrid_sim, 4),
        "true_positives": sorted(list(tp)),
        "false_positives": sorted(list(fp)),
        "false_negatives": sorted(list(fn))
    }
    return result

def create_index_pdf(keyword_pages_dict, output_path, book_title):
    """
    Membuat file PDF dari hasil indeks menggunakan ReportLab Platypus Table.
    Ini memperbaiki masalah layout tumpang tindih (overlap).
    """
    try:
        pdfmetrics.registerFont(TTFont('Helvetica', 'Helvetica.ttf'))
        pdfmetrics.registerFont(TTFont('Helvetica-Bold', 'Helvetica-Bold.ttf'))
    except:
        print("Peringatan: Tidak dapat mendaftarkan font Helvetica. Menggunakan font default.")

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    
    styles = getSampleStyleSheet()
    # Style untuk judul utama
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['h1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=10*mm
    )
    # Style untuk judul buku
    book_title_style = ParagraphStyle(
        'BookTitleStyle',
        parent=styles['h2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        alignment=TA_CENTER,
        spaceAfter=5*mm
    )
    # Style untuk huruf (A, B, C)
    letter_header_style = ParagraphStyle(
        'LetterHeader',
        parent=styles['h3'],
        fontName='Helvetica-Bold',
        fontSize=14,
        spaceBefore=5*mm,
        spaceAfter=2*mm
    )
    # Style untuk frasa (kolom kiri)
    phrase_style = ParagraphStyle(
        'Phrase',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        alignment=TA_LEFT
    )
    # Style untuk halaman (kolom kanan)
    page_style = ParagraphStyle(
        'Page',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        alignment=TA_RIGHT
    )

    story = []
    
    story.append(Paragraph("Indeks Buku Otomatis", title_style))
    story.append(Paragraph(book_title, book_title_style))

    # Urutkan frasa berdasarkan abjad, case-insensitive
    sorted_phrases = sorted(keyword_pages_dict.keys(), key=str.lower)
    
    current_letter = ""
    table_data = []
    table_styles_cmd = []

    for phrase in sorted_phrases:
        pages_list = keyword_pages_dict.get(phrase, [])
        # --- PERBAIKAN: Bungkus nomor halaman jika terlalu panjang ---
        # Kita akan membungkus secara manual dengan batas karakter (contoh: 50 char)
        raw_page_str = ", ".join(map(str, pages_list))
        wrapped_page_lines = wrap(raw_page_str, 50) # Bungkus setiap 50 karakter
        pages_str = "hlm.<br/>" + "<br/>".join(wrapped_page_lines)
        
        first_letter = phrase[0].upper()
        if first_letter != current_letter:
            current_letter = first_letter
            # Tambahkan baris untuk huruf header
            row_index = len(table_data)
            table_data.append([Paragraph(current_letter, letter_header_style), ''])
            # Tambahkan style untuk baris header
            table_styles_cmd.append(('SPAN', (0, row_index), (1, row_index))) # Gabungkan kolom
            table_styles_cmd.append(('TOPPADDING', (0, row_index), (0, row_index), 5*mm))
            
        # Tambahkan baris untuk frasa dan halaman
        table_data.append([
            Paragraph(phrase, phrase_style),
            Paragraph(pages_str, page_style)
        ])

    if not table_data:
        story.append(Paragraph("Indeks tidak menghasilkan data.", styles['Normal']))
        doc.build(story)
        print(f"PDF (kosong) berhasil dibuat di {output_path}")
        return

    # Tentukan lebar kolom (70% frasa, 30% halaman)
    total_width = doc.width
    col1_width = total_width * 0.7
    col2_width = total_width * 0.3

    # Buat tabel
    index_table = Table(table_data, colWidths=[col1_width, col2_width])
    
    # Tambahkan style dasar (garis tak terlihat, perataan)
    base_style = [
        ('GRID', (0,0), (-1,-1), 0.5, colors.transparent), # Garis tak terlihat
        ('BOX', (0,0), (-1,-1), 0.5, colors.transparent),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1*mm),
    ]
    
    # Gabungkan style dasar dengan style header huruf
    index_table.setStyle(TableStyle(base_style + table_styles_cmd))
    
    story.append(index_table)
    
    doc.build(story)
    print(f"PDF (Table Layout) berhasil dibuat di {output_path}")


# --- Rute Aplikasi Flask ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 1. Validasi Input
        if 'file_buku' not in request.files:
            flash('Tidak ada file buku yang diunggah', 'error')
            return redirect(request.url)
        
        file = request.files['file_buku']
        book_title = request.form.get('book_title', '').strip()
        extra_stopwords_str = request.form.get('extra_stopwords', '')
        
        if file.filename == '':
            flash('Tidak ada file yang dipilih', 'error')
            return redirect(request.url)
        
        if not book_title:
            flash('Judul buku wajib diisi', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            buku_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(buku_path)
            
            print(f"File buku disimpan di: {buku_path}")
            
            # === PERUBAHAN: Simpan path buku di sesi ===
            session['buku_path'] = buku_path
            # ==========================================
            
            # Buat set stopwords dinamis
            user_stopwords = set(re.split(r'[,\s\n]+', extra_stopwords_str.lower()))
            dynamic_stop_words_set = combined_stopwords.union(user_stopwords)

            # --- Jalankan Pipeline Lengkap ---
            
            # 1. Preprocess PDF
            print("Memulai preprocessing PDF...")
            page_texts, all_text, error = preprocess_pdf(buku_path, dynamic_stop_words_set)
            if error:
                flash(error, 'error')
                return redirect(request.url)
            print(f"Preprocessing selesai. {len(page_texts)} halaman diproses.")
            
            # (Simpan page_texts di sesi untuk pencarian cepat nanti? - Mungkin terlalu besar)
            # (Untuk saat ini, kita akan membaca ulang file)

            # 2. Ekstrak Bab
            print("Mengekstrak bab...")
            chapters = extract_chapters(buku_path)
            print(f"Ekstraksi bab selesai. {len(chapters)} bab ditemukan.")

            # 3. Ekstrak TF-IDF
            print("Mengekstrak TF-IDF...")
            tfidf_results = extract_keywords_tfidf_per_chapter(chapters, dynamic_stop_words_set, top_n=30)
            tfidf_results_stemmed = filter_stemmed_keywords(tfidf_results, stemmer)
            print("Ekstraksi TF-IDF selesai.")

            # 4. Ekstrak YAKE
            print("Mengekstrak YAKE...")
            keyphrases_yake, _ = run_yake_pipeline(all_text, page_texts, dynamic_stop_words_set)
            print("Ekstraksi YAKE selesai.")
            
            # 5. Bandingkan & Filter
            print("Membandingkan keyword dengan judul...")
            matched_keywords = compare_title_with_keywords(
                book_title, tfidf_results_stemmed, keyphrases_yake, w2v_model, threshold=0.569
            )
            # matched_keywords_filtered = filter_semantic_duplicates(matched_keywords)
            print("Perbandingan selesai.")

            # 6. Buat Indeks Final
            print("Membangun indeks final...")
            # keyword_pages_dict = build_final_index(matched_keywords_filtered, page_texts)
            keyword_pages_dict = build_final_index(matched_keywords, page_texts)
            print("Indeks final selesai.")
            
            # 7. Buat PDF Hasil
            # print("Membuat PDF hasil...")
            # pdf_filename = f"indeks_{filename}"
            # pdf_path = os.path.join(app.config['RESULT_FOLDER'], pdf_filename)
            # create_index_pdf(keyword_pages_dict, pdf_path, book_title)
            # print("PDF hasil selesai.")
            
            # 8. Simpan di Sesi
            session['book_title'] = book_title
            session['matched_keywords'] = matched_keywords # Simpan untuk evaluasi
            # session['matched_keywords'] = matched_keywords_filtered # Simpan untuk evaluasi
            session['keyword_pages'] = keyword_pages_dict
            session['download_file'] = filename
            
            return redirect(url_for('index'))

    # Metode GET: Tampilkan halaman
    book_title = session.get('book_title')
    results = session.get('keyword_pages')
    download_file = session.get('download_file')
    eval_results = session.get('eval_results')
    
    # Menggunakan render_template untuk file HTML eksternal
    return render_template("index.html",
                                  book_title=book_title,
                                  results=results,
                                  download_file=download_file,
                                  eval_results=eval_results)

# === BARU: API Endpoint untuk Pencarian Halaman ===
@app.route('/api/search_phrase', methods=['POST'])
def api_search_phrase():
    """
    API untuk mencari halaman (page numbers) dari sebuah frasa
    di dalam buku yang saat ini ada di sesi.
    """
    # 1. Cek sesi & data
    buku_path = session.get('buku_path')
    if not buku_path or not os.path.exists(buku_path):
        return jsonify({'status': 'error', 'message': 'Sesi buku tidak ditemukan. Harap unggah ulang file buku.'}), 400
    
    data = request.json
    phrase = data.get('phrase', '').strip()
    if not phrase:
        return jsonify({'status': 'error', 'message': 'Frasa tidak boleh kosong.'}), 400

    # 2. Buka PDF dan cari
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

    # 3. Kembalikan hasil
    if found_pages:
        return jsonify({'status': 'success', 'pages': found_pages})
    else:
        # Menggunakan pesan kustom Anda
        return jsonify({'status': 'not_found', 'message': 'Indeks tidak ada di buku ini.'})
# =================================================


@app.route('/evaluasi', methods=['POST'])
def evaluasi():
    """Route untuk Evaluasi F1 Score."""
    
    # 1. Ambil data dari sesi
    matched_keywords = session.get('matched_keywords')
    if not matched_keywords:
        flash('Sesi tidak ditemukan. Harap unggah file buku terlebih dahulu.', 'error')
        return redirect(url_for('index'))
        
    # 2. Validasi file upload
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
        
        # 3. Jalankan Evaluasi
        print("Memulai evaluasi...")
        if not w2v_model:
            flash('Model Word2Vec tidak dimuat, evaluasi hybrid similarity mungkin tidak akurat.', 'warning')
            
        eval_results = evaluasi_indeks(gt_path, matched_keywords, w2v_model)
        print("Evaluasi selesai.")
        
        if eval_results.get('error'):
            flash(eval_results['error'], 'error')
        
        # 4. Simpan hasil evaluasi di sesi
        session['eval_results'] = eval_results
        
    return redirect(url_for('index'))

# ... (Kode sebelumnya) ...

# === LOGIKA BARU: BULK ACTIONS ===

@app.route('/api/bulk_delete', methods=['POST'])
def api_bulk_delete():
    """
    Menghapus item dari sesi berdasarkan list frasa yang dikirim frontend.
    """
    try:
        data = request.json
        phrases_to_delete = data.get('phrases', [])
        
        # Ambil data saat ini dari session
        current_index = session.get('keyword_pages', {})
        
        if not current_index:
             return jsonify({'status': 'error', 'message': 'Data sesi kosong.'}), 400

        deleted_count = 0
        for phrase in phrases_to_delete:
            # Pastikan phrase ada sebelum delete (menghindari error key)
            if phrase in current_index:
                del current_index[phrase]
                deleted_count += 1
        
        # Simpan kembali ke session
        session['keyword_pages'] = current_index
        session.modified = True
        
        return jsonify({'status': 'success', 'deleted_count': deleted_count})
        
    except Exception as e:
        print(f"Error bulk delete: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/download_selected_pdf', methods=['POST'])
def api_download_selected_pdf():
    """
    Membuat dan mengirim PDF yang HANYA berisi frasa yang dipilih user.
    """
    try:
        data = request.json
        selected_phrases = data.get('phrases', [])
        
        if not selected_phrases:
            return jsonify({'status': 'error', 'message': 'Tidak ada frasa yang dipilih.'}), 400

        # Ambil data lengkap dari session
        full_index = session.get('keyword_pages', {})
        book_title = session.get('book_title', 'Indeks Terpilih')
        
        # Filter data: Hanya ambil yang dipilih user
        # Kita mengambil page numbers dari session (backend) agar formatnya tetap list[int]
        # bukan parsing string "hlm." dari frontend yang rentan error.
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
        
        # Gunakan fungsi create_index_pdf yang sudah ada
        # Kita tambahkan penanda di judul agar user tahu ini parsial
        create_index_pdf(filtered_data, pdf_path, f"{book_title} (Terpilih)")
        
        return send_file(pdf_path, as_attachment=True, download_name=pdf_filename)

    except Exception as e:
        print(f"Error download selected: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ... (Kode if __name__ == '__main__': ...)

@app.route('/download/<original_filename>')
def download(original_filename):
    """
    Route untuk mengunduh hasil indeks (PDF).
    PDF dibuat secara on-demand saat rute ini dipanggil.
    """
    # 1. Ambil data dari sesi
    keyword_pages = session.get('keyword_pages')
    book_title = session.get('book_title')
    
    if not keyword_pages or not book_title:
        flash('Sesi hasil indeks tidak ditemukan. Harap proses ulang file buku.', 'error')
        return redirect(url_for('index'))

    # 2. Tentukan nama dan path file PDF
    pdf_filename = f"indeks_{secure_filename(original_filename)}"
    pdf_path = os.path.join(app.config['RESULT_FOLDER'], pdf_filename)

    try:
        # 3. Buat PDF
        print(f"Membuat PDF on-demand di: {pdf_path}")
        create_index_pdf(keyword_pages, pdf_path, book_title)
        
        # 4. Kirim file
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
    # Pastikan path ke file .vec ada
    if not os.path.exists(model_path_vec) and not os.path.exists(model_path_gensim):
        print("-" * 50)
        print(f"PERINGATAN: File model '{model_path_vec}' atau '{model_path_gensim}' tidak ditemukan.")
        print(f"Harap unduh file model 'cc.id.300.vec' (atau .model) dan letakkan di folder: {os.getcwd()}")
        print("Anda dapat mengunduhnya dari: https://fasttext.cc/docs/en/crawl-vectors.html")
        print("-" * 50)
    
    app.run(debug=True) # Set debug=False untuk produksi