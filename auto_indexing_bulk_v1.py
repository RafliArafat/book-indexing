import os
import re
import numpy as np
from collections import Counter, defaultdict
import pandas as pd

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
import fasttext.util

# =============================================================================
# KONFIGURASI PATH
# =============================================================================

BASE_PATH   = r"C:\SKRIPSI (code)\DATASET INDEXING"
OUTPUT_DIR  = r"C:\SKRIPSI (code)\hasil_index_v1_bulk"
FASTTEXT_PATH = r'C:\SKRIPSI (code)\models\cc.id.300.bin'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# PEMETAAN JUDUL BUKU
# =============================================================================

pemetaan_judul = {
    "Buku 1":  "UTILITAS (Penyediaan Tenaga Listrik & Mekanik)",
    "Buku 2":  "MEKANIKA REKAYASA I",
    "Buku 3":  "BRANDING DESTINASI WISATA",
    "Buku 4":  "PRINSIP DAN PENERAPAN PROSES PERPINDAHAN PANAS",
    "Buku 5":  "APLIKASI KOMPUTER DASAR UNTUK BISNIS",
    "Buku 6":  "Manajemen Rantai Pasok",
    "Buku 7":  "Pengolahan Air Limbah: BIOFILTER-NANO",
    "Buku 8":  "KOREPONDENSI BISNIS",
    "Buku 9":  "Sistem Kendali Digital: Simulasi dan Implementasi Pada Mikrokontroler AVR",
    "Buku 10": "Pemanfaatan dan Peningkatan Nilai Kalori Batubara di Indonesia",
    "Buku 11": "PENGENDALIAN PROSES",
    "Buku 12": "Digitalisasi Akuntansi 1: Aplikasi ABSS",
    "Buku 13": "Contextual Approach pada CAD/CAM & Pemrograman CNC",
    "Buku 14": "Prinsip Dasar Keselamatan Kerja",
    "Buku 15": "PRAKTIK MATEMATIKA TERAPAN",
    "Buku 16": "PENGEMBANGAN TEACHING FACTORY (TEFA) BERBASIS ANALISIS SWOT",
    "Buku 17": "PROTOTIPE DARI KOMUNIKASI BISNIS",
    "Buku 18": "Motor Bakar 1",
    "Buku 19": "Desain Instalasi Listrik I & II",
    "Buku 20": "MACROMEDIA FLASH Untuk Media Pembelajaran Mahasiswa Vokasi",
    "Buku 21": "PETUNJUK PRAKTIKUM ANALISIS INSTRUMENTAL",
    "Buku 22": "Penerapan Reverse Engineering dalam Pembuatan Ankle-Foot Orthosis",
    "Buku 23": "OPTIMASI PROSES DAN APLIKASINYA DI BIDANG TEKNIK KIMIA",
    "Buku 24": "CUSTOMER RELATIONSHIP MANAGEMENT DAN PRAKTIK",
    "Buku 25": "TEKNIK MICROSOFT EXCEL PADA BIDANG AKUNTANSI DAN KEUANGAN",
    "Buku 26": "REKAMAN INFORMASI ARSIP BATIK BERBASIS WEB",
    "Buku 27": "SISTEM VIDEO DAN TELEVISI SIARAN",
    "Buku 28": "E-COMMERCE DAN PERILAKU KONSUMEN",
    "Buku 29": "Mengenal Algoritma dan Struktur Data dengan Panduan Praktis",
    "Buku 30": "PEMROGRAMAN WEB LANJUT (LARAVEL 10)",
    "Buku 31": "ASPEK HUKUM KEARSIPAN DAN INFORMASI",
    "Buku 32": "PERLINDUNGAN KEAMANAN DALAM INDUSTRI TELEKOMUNIKASI",
    "Buku 33": "BELAJAR DESAIN PROYEK MENGGUNAKAN JIRA",
    "Buku 34": "Sistem Pendukung Keputusan",
    "Buku 35": "Spreadsheet Untuk Perhitungan Termodinamika Teknik Kimia",
    "Buku 36": "ETIKA AKUNTANSI",
    "Buku 37": "AKUNTANSI PERPAJAKAN",
    "Buku 42": "Matematika Terapan I",
    "Buku 21 09.25.47": "MANAJEMEN SUMBER DAYA MANUSIA DAN PRAKTEK",
}

# =============================================================================
# RINGKASAN / ABSTRAK PER BUKU
# Isi kolom "ringkasan" di bawah ini.
# Minimal 50 karakter per buku agar FastText bisa bekerja dengan baik.
# Buku yang belum diisi akan di-skip dengan peringatan.
# =============================================================================

pemetaan_ringkasan = {
    "Buku 1":  "Prinsip pengadaan energi listrik, hukum dasar konversi energi elektromagnet, fenomena induksi elektromagnetik (Hukum Faraday), pembangkitan Gaya Gerak Listrik (GGL), serta konsep kopel (torsi). Generator Arus Searah (GDC), Konsep sistem tegangan listrik bolak-balik (AC) tiga fase, Konstruksi fisik dan prinsip kerja motor DC.",   # TODO: isi ringkasan Buku 1
    "Buku 2":  "",   # TODO: isi ringkasan Buku 2
    "Buku 3":  "",   # TODO: isi ringkasan Buku 3
    "Buku 4":  "",   # TODO: isi ringkasan Buku 4
    "Buku 5":  "",   # TODO: isi ringkasan Buku 5
    "Buku 6":  "",   # TODO: isi ringkasan Buku 6
    "Buku 7":  "",   # TODO: isi ringkasan Buku 7
    "Buku 8":  "",   # TODO: isi ringkasan Buku 8
    "Buku 9":  "",   # TODO: isi ringkasan Buku 9
    "Buku 10": "",   # TODO: isi ringkasan Buku 10
    "Buku 11": "Konsep dasar sistem kontrol, peran penting instrumentasi, serta definisi proses dan sistem dinamis, perangkat keras kontrol mencakup sensor, transmitter, katup kontrol pengembangan flowsheet. Penyelesaian Persamaan Diferensial Biasa (ODE). Karakteristik respons sistem loop terbuka vs loop tertutup, penggunaan variabel deviasi, analisis respons frekuensi (Plot Bode), Transformasi Laplace dan Fungsi Transfer, representasi Diagram Blok, overdamped, critically damped, underdamped, tidak stabil, dan inverse response.",   # TODO: isi ringkasan Buku 11
    "Buku 12": "",   # TODO: isi ringkasan Buku 12
    "Buku 13": "Konsep dasar teknologi Computer-Aided Design (CAD) dan Computer-Aided Manufacturing (CAM). perangkat lunak Mastercam, Pembuatan desain sketsa 2D (garis, chamfer, fillet), simulasi proses pembubutan yang mencakup roughing, finishing, threading (ulir), grooving (alur), dan cutoff (pemotongan). Dasar-proses milling, pembuatan desain 2D untuk milling dan mesin CNC.",   # TODO: isi ringkasan Buku 13
    "Buku 14": "",   # TODO: isi ringkasan Buku 14
    "Buku 15": "",   # TODO: isi ringkasan Buku 15
    "Buku 16": "Konsep, bentuk, dan tahapan penyusunan Standar Operasional Prosedur (SOP) yang berfokus pada SOP Model Manajemen. Matriks SWOT berupa Strategi SO (Agresif), Strategi ST (Diversifikasi), Strategi WO (Turn-around), dan Strategi WT (Defensif). Implementasi dan perhitungan matriks Internal Factor Analysis Summary (IFAS) dan External Factor Analysis Summary (EFAS). Penentuan posisi strategi dan arah kebijakan TEFA melalui Grafik Kuadran.",   # TODO: isi ringkasan Buku 16
    "Buku 17": "",   # TODO: isi ringkasan Buku 17
    "Buku 18": "",   # TODO: isi ringkasan Buku 18
    "Buku 19": "",   # TODO: isi ringkasan Buku 19
    "Buku 20": "",   # TODO: isi ringkasan Buku 20
    "Buku 21": "",   # TODO: isi ringkasan Buku 21
    "Buku 22": "perancangan serta manufaktur perangkat medis Ankle-Foot Orthosis (AFO). penerapan Additive Manufacturing (3D Printing), Pengenalan teknologi Reverse Engineering. Manfaat penggunaan AFO untuk rehabilitasi pasien (seperti penderita cedera pergelangan kaki atau foot drop). pemindaian 3D (3D scanning), pemodelan CAD, analisis elemen hingga (finite element).",   # TODO: isi ringkasan Buku 22
    "Buku 23": "Konsep dasar optimasi, ruang lingkup, struktur hierarki, Dasar-dasar program MATLAB, pengoperasian fungsi matematika, manajemen File dan M-File, penggunaan struktur kontrol Loops, Conditional Statements, Switch-Case degree of freedom. algoritma Metode Newton-Raphson, Metode Simpson’s 1/3 Rule, Pengertian persamaan diferensial, penyelesaian Persamaan Diferensial Biasa (PDB) Metode Runge-Kutta",   # TODO: isi ringkasan Buku 23
    "Buku 24": "",   # TODO: isi ringkasan Buku 24
    "Buku 25": "",   # TODO: isi ringkasan Buku 25
    "Buku 26": "",   # TODO: isi ringkasan Buku 26
    "Buku 27": "",   # TODO: isi ringkasan Buku 27
    "Buku 28": "",   # TODO: isi ringkasan Buku 28
    "Buku 29": "",   # TODO: isi ringkasan Buku 29
    "Buku 30": "",   # TODO: isi ringkasan Buku 30
    "Buku 31": "dasar-dasar Hukum Pidana dan Hukum Perdata, Konsep dasar dan pengertian arsip, tujuan utama pengelolaan kearsipan, pengenalan awal mengenai pengertian legalisasi dokumen, carding cracking cyberstalking, ubungan antara legalisasi dokumen dengan ketercapaian tujuan arsip, pemahaman mendalam mengenai prosedur legalisasi, roses transformasi digitalisasi arsip, jaminan autentisitas dan legalitas hukum arsip hasil digitalisasi, yuridis serta analisis hukum pada studi kasus penerapan Aplikasi Srikandi.",   # TODO: isi ringkasan Buku 31
    "Buku 32": "",   # TODO: isi ringkasan Buku 32
    "Buku 33": "engineering project design, Pengertian dan filosofi dasar dari metodologi Agile, implementasi Agile, kerangka kerja Scrum, seperti Product Owner, Scrum Master, dan Development Team, seperti Sprint Planning, Daily Standup, Sprint Review, dan Retrospective, Pengenalan JIRA sebagai alat bantu (tool) manajemen proyek berbasis Agile/Scrum, Manajemen backlog dan siklus kerja di JIRA, yang mencakup pembuatan dan pengaturan Sprint, pengelompokan fitur besar melalui Epic Project Pages.",   # TODO: isi ringkasan Buku 33
    "Buku 34": "menjadikan mahasiswa mampu memahami konsep, teknologi, model, dan aplikasi SPK, Metode Weighted Sum Model (WSM), Metode Weighted Product Model (WPM), Metode Analytic Hierarchy Process (AHP), Metode Elimination Et Choix Traduisant la Réalité (Electre), Metode Evaluation Based On Distance From Average Solution (EDAS), Metode Multi-Attributive Border Approximation Area Comparison (MABAC), Konsep Group Decision Support System  (GDSS), dan Metode Fuzzy.",   # TODO: isi ringkasan Buku 34
    "Buku 35": "penyusunan neraca energi untuk sistem tertutup maupun sistem terbuka (steady-state flow system), Fenomena tekanan uap zat murni, Persamaan Antoine dan Persamaan Wagner, Karakteristik gas ideal, penyimpangan gas nyata melalui Persamaan Virial, penerapan berbagai Persamaan Keadaan (EOS - Equation of State), kesetimbangan fase uap-cair, pemodelan VLE sederhana menggunakan Hukum Raoult untuk menghitung titik embun (dewpoint) dan titik gelembung (bubblepoint), Persamaan Margules, Wilson, NRTL, dan UNIQUAC, Kesetimbangan Cair-Cair / LLE.",   # TODO: isi ringkasan Buku 35
    "Buku 36": "",   # TODO: isi ringkasan Buku 36
    "Buku 37": "",   # TODO: isi ringkasan Buku 37
    "Buku 42": "",   # TODO: isi ringkasan Buku 42
    "Buku 21 09.25.47": "",  # TODO: isi ringkasan Buku 21 09.25.47
}

# =============================================================================
# INISIALISASI NLP & STOPWORDS
# =============================================================================

print("Memuat stopwords dan model...")
try:
    stopwords.words('indonesian')
except LookupError:
    nltk.download('stopwords', quiet=True)

factory = StopWordRemoverFactory()
sastrawi_stop = set(factory.get_stop_words())
nltk_stop_id  = set(stopwords.words("indonesian"))
nltk_stop_en  = set(stopwords.words("english"))

extra_stopwords_global = {
    "pengantar", "pendahuluan", "bab", "daftar", "pustaka", "referensi",
    "abstrak", "kata", "modul", "ajar", "mata", "kuliah", "dan", "atau",
    "bagian", "pasal", "buku", "tujuan", "joko", "jurusan",
    "budi", "santi", "pada", "dalam", "untuk", "adalah", "yang", "oleh",
    "materi", "teknologi", "malang", "politeknik", "negeri", "rajin",
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"
}
combined_stopwords = (
    sastrawi_stop
    .union(nltk_stop_id)
    .union(nltk_stop_en)
    .union(extra_stopwords_global)
)

print("Memuat stemmer Sastrawi...")
stemmer = StemmerFactory().create_stemmer()

print("Memuat model FastText...")
ft_model = None
try:
    if os.path.exists(FASTTEXT_PATH):
        ft_model = fasttext.load_model(FASTTEXT_PATH)
        print(f"FastText berhasil dimuat. Dimensi: {ft_model.get_dimension()}")
    else:
        print(f"PERINGATAN: Model FastText tidak ditemukan di {FASTTEXT_PATH}")
except Exception as e:
    print(f"Gagal memuat FastText: {e}")

# =============================================================================
# FUNGSI PIPELINE (identik dengan auto_indexing.py)
# =============================================================================

def normalize_line(line):
    line = line.lower().strip()
    line = re.sub(r"\d+", "", line)
    return re.sub(r"\s+", " ", line)

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

def preprocess_pdf(pdf_path, dynamic_stopwords):
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
                        if i % 2 == 0: header_even.append(h)
                        else: header_odd.append(h)
                        footer_all.append(f)

        def common_set(lst, threshold=0.3):
            if not lst: return set()
            freq = Counter(lst)
            n = len(lst)
            return {h for h, c in freq.items() if c / n > threshold and h}

        common_even   = common_set(header_even)
        common_odd    = common_set(header_odd)
        common_footer = common_set(footer_all)

        page_texts = []
        all_text   = ""
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text()
                if raw:
                    lines = raw.split("\n")
                    if len(lines) > 2:
                        h = normalize_line(lines[0])
                        f = normalize_line(lines[-1])
                        if i % 2 == 0 and h in common_even:   lines = lines[1:]
                        if i % 2 == 1 and h in common_odd:    lines = lines[1:]
                        if f in common_footer:                 lines = lines[:-1]
                    raw   = "\n".join(lines)
                    clean = clean_text(raw)
                    # hapus stopwords dinamis (sama seperti auto_indexing.py)
                    words  = clean.split()
                    filtered = [w for w in words if w.lower() not in dynamic_stopwords]
                    clean_filtered = " ".join(filtered)
                    page_texts.append((i, clean_filtered))
                    all_text += clean_filtered + " "
        return page_texts, all_text.strip(), None
    except Exception as e:
        return [], "", f"Error membaca PDF: {e}"

def normalize_text(text):
    return re.sub(r'[^a-z0-9\s]', ' ', text.lower()).strip()

def phrase_embedding_fasttext(phrase, ft_model):
    if ft_model is None: return None
    tokens = normalize_text(phrase).split()
    if not tokens: return None
    return np.mean([ft_model.get_word_vector(t) for t in tokens], axis=0)

def extract_acronyms(text):
    pattern = r'([A-Za-z][A-Za-z\s\-]+)\s*\(([A-Z]{2,})\)'
    matches  = re.findall(pattern, text)
    return {short.lower(): long.lower().strip() for long, short in matches}

def expand_acronym_if_needed(phrase, acronym_map):
    return acronym_map.get(phrase.lower().strip(), phrase.lower().strip())

def compare_context_with_keywords_fasttext(
    book_title, book_summary, keywords, ft_model, threshold=0.3
):
    """
    Alur FastText dari auto_indexing.py:
    judul + ringkasan → satu embedding konteks → bandingkan tiap keyword.
    """
    if ft_model is None:
        return [(kw, 1.0, "no-filter") for kw in keywords]

    book_context = f"{book_title} {book_summary}"
    acronym_map  = extract_acronyms(book_context)
    context_vec  = phrase_embedding_fasttext(book_context, ft_model)

    if context_vec is None:
        return [(kw, 1.0, "no-filter") for kw in keywords]

    matched = []
    for kw in keywords:
        kw_expanded = expand_acronym_if_needed(kw, acronym_map)
        kw_vec = phrase_embedding_fasttext(kw_expanded, ft_model)
        if kw_vec is None: continue
        sim = 1 - cosine(context_vec, kw_vec)
        if sim >= threshold:
            matched.append((kw, float(sim), "fasttext-context"))

    return sorted(matched, key=lambda x: x[1], reverse=True)

def tokenize3(text):
    return set(re.findall(r"\w+", text.lower()))

def map_keywords_to_pages(matched_keywords, page_texts, overlap_threshold=0.8):
    matched_phrases = [kw for kw, _, _ in matched_keywords]
    keyword_pages   = defaultdict(list)
    for phrase in matched_phrases:
        phrase_tokens = tokenize3(phrase)
        if not phrase_tokens: continue
        for page_num, page_txt in page_texts:
            page_tokens = tokenize3(page_txt)
            if not page_tokens: continue
            overlap = len(phrase_tokens & page_tokens) / len(phrase_tokens)
            if overlap >= overlap_threshold or phrase.lower() in page_txt.lower():
                if page_num not in keyword_pages[phrase]:
                    keyword_pages[phrase].append(page_num)
    for phrase in keyword_pages:
        keyword_pages[phrase].sort()
    return dict(keyword_pages)

# --- YAKE helpers (identik dengan auto_indexing.py) ---

def merge_reversed_phrases(keywords):
    groups = {}
    for kw, score in keywords:
        if not kw or not kw.strip(): continue
        key = tuple(sorted([w.lower() for w in kw.split() if w]))
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
        no_consec = []
        for w in words:
            if not no_consec or w != no_consec[-1]: no_consec.append(w)
        seen, unique = set(), []
        for w in no_consec:
            if w.lower() not in seen:
                seen.add(w.lower())
                unique.append(w)
        cleaned.append((" ".join(unique), score))
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
        key = kw.lower()
        if key not in grouped or score < grouped[key][1]:
            grouped[key] = (kw, score)
    final = []
    for key_lower, (kw, score) in grouped.items():
        if any(k.upper() == k and k.lower() == key_lower for k, _ in keywords):
            if kw.upper() == kw:
                final.append((kw, score))
        else:
            final.append((kw, score))
    return final

def filter_phrases_with_existing_acronyms(keywords):
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
            if any(u in single_upper and i != 0 for i, u in enumerate(upper_words)):
                continue
        filtered.append((kw, score))
    return filtered

def extract_initials_from_title_phrase(phrase):
    tokens   = re.findall(r"\w+", phrase)
    initials = [t[0].upper() for t in tokens if t and (t[0].isupper() or t.isupper())]
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
            if normalize_acronym(initials) in acronyms:
                new_score = score * boost_factor
        boosted.append((kw, new_score))
    return boosted

def filter_similar_phrases_by_overlap_safe(matched_keywords, min_common=2, min_overlap_ratio=0.6):
    neutral_words = {
        "system", "sistem", "decision", "support", "sum", "product",
        "process", "fuzzy", "mabac", "ahp", "edas", "wsm", "multi-criteria", "making"
    }
    kw_lower = [(kw.lower(), score, kw) for kw, score in matched_keywords]
    kw_lower = sorted(kw_lower, key=lambda x: x[1])
    to_remove = set()
    for i, (kw1l, score1, _) in enumerate(kw_lower):
        if i in to_remove: continue
        words1 = set(kw1l.split())
        for j, (kw2l, score2, _) in enumerate(kw_lower[i + 1:], start=i + 1):
            if j in to_remove: continue
            words2  = set(kw2l.split())
            common  = words1 & words2
            if common and all(w in neutral_words for w in common): continue
            overlap_ratio = len(common) / min(len(words1), len(words2)) if min(len(words1), len(words2)) > 0 else 0
            if len(common) >= min_common and overlap_ratio >= min_overlap_ratio and common != {"weighted", "model"}:
                if len(words2) > len(words1) or score2 < score1: to_remove.add(i)
                else: to_remove.add(j)
    return sorted(
        [(orig_kw, s) for idx, (_, s, orig_kw) in enumerate(kw_lower) if idx not in to_remove],
        key=lambda x: x[1]
    )

def build_page_map(valid_phrases, page_texts, threshold=0.8):
    page_map = defaultdict(list)
    for phrase in valid_phrases:
        phrase_tokens = tokenize3(phrase)
        if not phrase_tokens: continue
        for page_num, page_txt in page_texts:
            page_tokens = tokenize3(page_txt)
            overlap = len(phrase_tokens & page_tokens) / len(phrase_tokens)
            if overlap >= threshold:
                page_map[phrase].append(page_num)
    return page_map

def merge_by_common_sequence(words1, words2):
    best_merge, max_common_len = None, 0
    for i in range(len(words1)):
        for j in range(len(words2)):
            k = 0
            while i + k < len(words1) and j + k < len(words2) and words1[i + k] == words2[j + k]:
                k += 1
            if k >= 2 and k > max_common_len:
                max_common_len = k
                if i == 0 and j > 0:   merged = words2[:j] + words1
                elif j == 0 and i > 0: merged = words1[:i] + words2
                elif j > 0 and i > 0:  continue
                else: merged = words1[:i] + words2[j:]
                best_merge = merged
    return best_merge

def merge_related_phrases(matched_keywords, page_map, page_texts, all_text_lower, overlap_ratio=0.6):
    def find_phrase_pages(phrase):
        return [pn for pn, txt in page_texts if phrase.lower() in txt.lower()]

    merged, used = [], set()
    original_case_map = {kw.lower(): kw for kw, _ in matched_keywords}

    for i, (kw1_orig, score1) in enumerate(matched_keywords):
        kw1 = kw1_orig.lower()
        if kw1 in used: continue
        words1 = kw1.split()
        if len(words1) < 3: continue
        for kw2_orig, score2 in matched_keywords[i + 1:]:
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
            pages3 = set(page_map.get(original_case_map.get(merged_phrase_lower, merged_phrase_lower), []))
            if not pages3: pages3 = set(find_phrase_pages(merged_phrase_lower))
            union_pages = pages1 | pages2
            if not union_pages: continue
            if len(pages3 & union_pages) / len(union_pages) >= overlap_ratio:
                merged_phrase_orig = original_case_map.get(merged_phrase_lower, merged_phrase_lower.title())
                merged.append((merged_phrase_orig, min(score1, score2)))
                used.add(kw1)
                used.add(kw2)

    merged_lower = {kw.lower() for kw, _ in merged}
    for kw_orig, score in matched_keywords:
        if kw_orig.lower() not in merged_lower and kw_orig.lower() not in used:
            merged.append((kw_orig, score))
    return merged

def merge_reversed_phrases_second(phrases):
    normalized = {}
    for ph in phrases:
        key = tuple(sorted(ph.split()))
        if key not in normalized: normalized[key] = ph
    return list(normalized.values())

def normalize_repeated_words_second(phrases):
    cleaned = []
    for ph in phrases:
        words = ph.split()
        no_consec = []
        for w in words:
            if not no_consec or w != no_consec[-1]: no_consec.append(w)
        seen, unique = set(), []
        for w in no_consec:
            if w.lower() not in seen:
                seen.add(w.lower())
                unique.append(w)
        cleaned.append(" ".join(unique))
    return cleaned

def run_yake_pipeline(all_text, page_texts, combined_stopwords_set, top_per_n=70):
    """
    Pipeline YAKE identik dengan auto_indexing.py.
    """
    keywords_all = []
    for n in [1, 2, 3]:
        kw_extractor = yake.KeywordExtractor(lan="multilingual", n=n, top=top_per_n)
        keywords_all.extend(kw_extractor.extract_keywords(all_text))

    seen, cleaned = set(), []
    for kw, score in sorted(keywords_all, key=lambda x: x[1]):
        kw = kw.strip()
        if not kw: continue
        words = kw.split()
        if len(words) > 1 and len(set(words)) == 1: continue
        if any(w.lower() in combined_stopwords_set for w in words): continue
        norm = kw.lower()
        if norm not in seen:
            seen.add(norm)
            cleaned.append((kw, score))
    cleaned = sorted(cleaned, key=lambda x: x[1])

    cleaned = normalize_repeated_words(cleaned)
    cleaned = merge_reversed_phrases(cleaned)
    cleaned = filter_multiword_capitalized_phrases(cleaned)
    cleaned = filter_by_capitalization(cleaned)
    cleaned = filter_phrases_with_existing_acronyms(cleaned)
    cleaned = boost_full_phrases_from_acronyms_v2(cleaned, boost_factor=0.7)
    cleaned = filter_similar_phrases_by_overlap_safe(cleaned, min_common=2)

    page_map_yake = build_page_map([kw for kw, _ in cleaned], page_texts)
    appended      = merge_related_phrases(cleaned, page_map_yake, page_texts, all_text.lower())

    keyphrases = [kw for kw, _ in appended]
    keyphrases = normalize_repeated_words_second(keyphrases)
    keyphrases = merge_reversed_phrases_second(keyphrases)
    return keyphrases, appended

# --- Evaluasi ---

def preprocess_phrases(phrases):
    roman   = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}
    cleaned = set()
    for ph in phrases:
        ph = re.sub(r"\(.*?\)", "", ph)
        ph = re.sub(r"[^\w\s]", " ", ph)
        ph = re.sub(r"\s+", " ", ph).lower().strip()
        if any(c.isalpha() for c in ph) and len(ph) > 1 and ph not in roman and not re.fullmatch(r"[a-zA-Z]", ph):
            cleaned.add(ph)
    return cleaned

def extract_keywords_from_index_file(gt_path):
    doc      = fitz.open(gt_path)
    keywords = set()
    for page in doc:
        text  = page.get_text("text")
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        merged_lines, buffer = [], ""

        def is_complete_entry(line): return bool(re.search(r'[·,:]\s*[ivxIVX\d]', line))

        for line in lines:
            if re.match(r'^[A-Z]$', line): continue
            if buffer:
                if not is_complete_entry(buffer) and not re.match(r'^[A-Z]\s*$', line):
                    buffer += " " + line
                    if is_complete_entry(line):
                        merged_lines.append(buffer)
                        buffer = ""
                    continue
                else:
                    merged_lines.append(buffer)
                    buffer = ""
            if not is_complete_entry(line) and not re.match(r'^[A-Z]$', line):
                buffer = line
            else:
                merged_lines.append(line)
        if buffer:
            merged_lines.append(buffer)

        for line in merged_lines:
            match = re.match(r"([A-Za-z\s\-\(\)]+?)\s*[·,:]", line.strip())
            # match = re.match(r"([A-Za-z\s\-\(\)]+)[,\s0-9ivxIVX]*$", line.strip())
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
            if gt in matched_gt: continue
            t     = 60 if len(gen) <= 3 else (75 if len(gen) <= 4 else threshold)
            score = fuzz.ratio(gen.lower(), gt.lower())
            if score >= t and score > best_score:
                best_score = score
                best_match = gt
        if best_match:
            tp.add(gen)
            matched_gt.add(best_match)
        else:
            fp.add(gen)
    return tp, fp, ground_truth - matched_gt

def fasttext_embed(text, model):
    tokens = text.lower().split()
    if not tokens or model is None: return np.zeros(300)
    return np.mean([model.get_word_vector(t) for t in tokens], axis=0)

def evaluasi_indeks(gt_path, matched_keywords, ft_model=None, use_fuzzy=True):
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    generated    = preprocess_phrases([kw for kw, _, _ in matched_keywords])
    ground_truth = extract_keywords_from_index_file(gt_path)

    if use_fuzzy: tp, fp, fn = fuzzy_match_evaluation(generated, ground_truth)
    else:
        tp = generated & ground_truth
        fp = generated - ground_truth
        fn = ground_truth - generated

    precision = len(tp) / len(generated)    if generated    else 0
    recall    = len(tp) / len(ground_truth) if ground_truth else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    ft_sim = None
    if ft_model and generated and ground_truth:
        emb1   = np.array([fasttext_embed(p, ft_model) for p in generated])
        emb2   = np.array([fasttext_embed(p, ft_model) for p in ground_truth])
        sims   = sk_cosine(emb1, emb2)
        ft_sim = round(float(sims.max(axis=1).mean()), 4)

    return {
        "precision":   round(precision, 4),
        "recall":      round(recall, 4),
        "f1_score":    round(f1, 4),
        "similarity":  ft_sim,
        "true_positives":  sorted(tp),
        "false_positives": sorted(fp),
        "false_negatives": sorted(fn),
    }

# =============================================================================
# MAIN BULK PIPELINE
# =============================================================================

if __name__ == "__main__":
    print(f"\nMemulai Auto Indexing Bulk (Versi v1 - pakai ringkasan manual)...")
    print(f"Dataset : {BASE_PATH}")
    print(f"Output  : {OUTPUT_DIR}\n")

    all_excel_rows = []
    folders = sorted(
        [f for f in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, f))]
    )

    for folder in folders:
        folder_path = os.path.join(BASE_PATH, folder)
        print(f"\n>>> Folder: {folder}")

        # --- Cari file PDF isi & indeks ---
        pdf_files  = os.listdir(folder_path)
        isi_pdf    = None
        indeks_pdf = None
        for file in pdf_files:
            if file.lower().endswith(".pdf"):
                if "indeks" in file.lower(): indeks_pdf = os.path.join(folder_path, file)
                else:                        isi_pdf    = os.path.join(folder_path, file)

        output_file = os.path.join(OUTPUT_DIR, f"{folder}.txt")

        try:
            if isi_pdf is None:
                raise Exception("File isi PDF tidak ditemukan.")

            # --- Judul & Ringkasan ---
            book_title   = pemetaan_judul.get(folder, folder)
            book_summary = pemetaan_ringkasan.get(folder, "").strip()

            print(f"  -> Judul    : {book_title}")

            if not book_summary or len(book_summary) < 50:
                print(f"  -> SKIP: Ringkasan '{folder}' belum diisi (minimal 50 karakter).")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write("STATUS: SKIP\n\n")
                    f.write(f"Ringkasan untuk '{folder}' belum diisi di pemetaan_ringkasan.\n")
                continue

            print(f"  -> Ringkasan: {book_summary[:80]}...")

            # --- 1. Preprocess PDF ---
            print("  -> Preprocessing PDF...")
            page_texts, all_text, error = preprocess_pdf(isi_pdf, combined_stopwords)
            if error:
                raise Exception(error)

            # --- 2. YAKE Pipeline ---
            print("  -> YAKE Pipeline...")
            keyphrases, keyphrases_tuples = run_yake_pipeline(
                all_text, page_texts, combined_stopwords, top_per_n=100
            )
            print(f"     Keyword hasil YAKE: {len(keyphrases)}")

            # --- 3. FastText Filter (alur v1: judul+ringkasan → satu vektor konteks) ---
            print("  -> FastText Filtering...")
            matched_keywords = compare_context_with_keywords_fasttext(
                book_title   = book_title,
                book_summary = book_summary,
                keywords     = keyphrases,
                ft_model     = ft_model,
                threshold    = 0.3
            )
            print(f"     Keyword matched: {len(matched_keywords)}")

            # --- 4. Pemetaan Halaman ---
            keyword_pages = map_keywords_to_pages(matched_keywords, page_texts)

            # --- 5. Evaluasi (jika ada ground truth) ---
            f1, precision, recall, ft_sim = 0.0, 0.0, 0.0, None
            if indeks_pdf:
                print("  -> Evaluasi ground truth...")
                eval_results = evaluasi_indeks(indeks_pdf, matched_keywords, ft_model, use_fuzzy=True)
                f1        = eval_results["f1_score"]
                precision = eval_results["precision"]
                recall    = eval_results["recall"]
                ft_sim    = eval_results["similarity"]
                print(f"     F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}")
            else:
                print("  -> Tidak ada PDF indeks ground truth.")

            # --- 6. Tulis Laporan TXT ---
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("STATUS: SUCCESS\n\n")
                f.write(f"Judul Buku  : {book_title}\n")
                f.write(f"F1 Score    : {f1:.4f}\n")
                f.write(f"Precision   : {precision:.4f}\n")
                f.write(f"Recall      : {recall:.4f}\n")
                if ft_sim is not None:
                    f.write(f"FT Similarity: {ft_sim:.4f}\n")
                f.write("\n" + "=" * 50 + "\n\n")
                for kw, pages in keyword_pages.items():
                    pages_str = ", ".join(map(str, pages))
                    f.write(f"{kw} | {pages_str}\n")
                    all_excel_rows.append({
                        "Folder":     folder,
                        "Judul Buku": book_title,
                        "Keyword":    kw,
                        "Pages":      pages_str,
                        "F1 Score":   f1,
                        "Precision":  precision,
                        "Recall":     recall,
                        "FT Sim":     ft_sim,
                    })

            print(f"  -> Selesai. Hasil: {output_file}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  -> ERROR: {e}")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("STATUS: ERROR\n\n")
                f.write(str(e))

    # --- Export Excel ---
    if all_excel_rows:
        df         = pd.DataFrame(all_excel_rows)
        excel_path = os.path.join(OUTPUT_DIR, "rekap_semua_buku_v1_bulk_indeks.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"\n{'='*55}")
        print(f"Selesai! Rekap Excel tersimpan di:\n{excel_path}")
        print(f"{'='*55}")
    else:
        print("\nTidak ada data yang berhasil diproses.")
