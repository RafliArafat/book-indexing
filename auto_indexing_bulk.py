import os
import re
import math
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

# --- Summarizer (sumy) ---
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# =============================================================================
# KONFIGURASI PATH
# =============================================================================
# Path folder dataset Anda
BASE_PATH = r"C:\SKRIPSI (code)\DATASET INDEXING"

# Path output BARU (dibedakan dari sebelumnya)
OUTPUT_DIR = r"C:\SKRIPSI (code)\hasil_index_v3"

# Path model FastText bin
FASTTEXT_PATH = r'C:\SKRIPSI (code)\models\cc.id.300.bin'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# DICTIONARY PEMETAAN JUDUL
# Silakan isi judul lengkapnya di sini sesuai nama foldernya
# ---------------------------------------------------------
pemetaan_judul = {
    "Buku 1": "UTILITAS (Penyediaan Tenaga Listrik & Mekanik)",
        "Buku 2": "MEKANIKA REKAYASA I ",
        "Buku 3": "BRANDING DESTINASI WISATA",
        "Buku 4": "PRINSIP DAN PENERAPAN PROSES PERPINDAHAN PANAS",
        "Buku 5": "APLIKASI KOMPUTER DASAR UNTUK BISNIS",
        "Buku 6": "Manajemen Rantai Pasok",
        "Buku 7": "Pengolahan Air Limbah: BIOFILTER-NANO",
        "Buku 8": "KOREPONDENSI BISNIS",
        "Buku 9": "Sistem Kendali Digital: Simulasi dan Implementasi Pada Mikrokontroler AVR",
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
        "Buku 22": "Penerapan Reverse Engineering dalam Pembuatan Ankle-Foot Orthosis ",
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
        "Buku 21 09.25.47": "MANAJEMEN SUMBER DAYA MANUSIA DAN PRAKTEK"
}

# =============================================================================
# INISIALISASI NLP & STOPWORDS
# =============================================================================
print("Memuat stopwords dan model...")
try:
    stopwords.words('indonesian')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)

STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk", "dalam", "adalah", "pada", "atau", "tidak", "juga", "akan", "dapat", "oleh", "sebagai", "ada", "telah", "maka", "sehingga", "agar", "bagi", "karena", "seperti", "antara", "setiap", "suatu", "berdasarkan", "tersebut", "dimana", "bahwa", "adapun", "yaitu", "yakni", "masing", "tiap", "setelah", "sebelum", "kemudian", "selanjutnya", "pertama", "kedua", "ketiga", "langkah", "cara", "berikut", "berikutnya", "lebih", "sangat", "hanya", "jika", "apabila", "hingga", "sampai", "melalui", "terhadap", "jadi", "sudah", "pun", "hal", "namun", "tetapi", "tapi", "walau", "meski", "meskipun", "walaupun", "bisa", "boleh", "harus", "perlu", "ingin", "lain", "lainnya", "serta", "maupun", "baik", "bukan", "belum", "masih", "paling", "nilai", "skor", "menggunakan", "digunakan", "dilakukan", "dihitung", "diperoleh", "didapatkan", "ditentukan", "dibuat", "terdiri", "terbentuk", "merupakan", "memiliki", "mempunyai", "dihasilkan", "diberikan", "disebut", "dikenal", "salah", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan", "sepuluh", "semua", "seluruh", "sebuah", "beberapa", "banyak", "kasus", "tabel", "gambar", "bab", "persamaan", "keterangan", "dst", "dll", "dsb", "lampiran", "hlm", "hal", "cet", "edisi", "cetakan", "penerbit", "pengarang", "penulis", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "melainkan", "melakukan", "membuat", "menjadi", "menyebabkan", "menjelaskan", "menunjukkan", "menghasilkan", "menentukan", "mendapatkan", "menghitung", "memperoleh", "meningkatkan", "menurunkan", "mengurangi", "menambah", "membantu", "mencapai", "melihat", "mengetahui", "memahami", "membutuhkan", "memerlukan", "mempengaruhi", "menggambarkan", "menyatakan", "menerapkan", "memberikan", "melibatkan", "mengambil", "mengeluarkan", "menyusun", "membentuk", "mengandung", "mengubah", "menyediakan", "mengelola", "mengidentifikasi", "mengevaluasi", "menganalisis", "dikembangkan", "diterapkan", "dianalisis", "dievaluasi", "diidentifikasi", "dikelola", "diubah", "disusun", "dibentuk", "diambil", "disediakan", "diukur", "dipilih", "ditetapkan", "dijelaskan", "ditunjukkan", "digambarkan", "dinyatakan", "disebutkan", "dikelompokkan", "dibandingkan", "diperhitungkan", "dijadikan", "diasumsikan", "terbaik", "terpilih", "terakhir", "terbaru", "tertentu", "terdapat", "terlihat", "terutama", "tergantung", "termasuk", "terjadi", "terkait", "terkecuali", "berbeda", "berkaitan", "berhubungan", "berpengaruh", "berupa", "berjumlah", "berada", "berlaku", "berfungsi", "berperan", "berkembang", "bertujuan", "bersifat", "berasal", "berdasar", "bergantung", "seharusnya", "sebaiknya", "kadang", "kadangkala", "sering", "selalu", "jarang", "biasanya", "umumnya", "akhirnya", "awalnya", "sebenarnya", "sesungguhnya", "tentunya", "tentu", "memang", "justru", "bahkan", "oleh karena", "setidaknya", "setidak", "sekurangnya", "paling tidak", "sekaligus", "sekarang", "saat ini", "saat", "tersebut di atas", "berikut ini", "sebagai berikut", "antara lain", "dan lain", "dan sebagainya", "misalnya", "contohnya", "artinya", "maksudnya", "dimaksud", "disini", "supaya", "buruk", "tinggi", "rendah", "besar", "kecil", "penting", "relevan", "efektif", "efisien", "optimal", "akurat", "valid", "reliabel", "objektif", "pendahuluan", "penutup", "kesimpulan", "saran", "daftar", "pustaka", "referensi", "abstrak", "kata pengantar", "daftar isi", "daftar tabel", "daftar gambar", "proses", "teknik", "prosedur", "tahap", "langkah", "aspek", "jenis", "bentuk", "tipe", "kategori", "kelompok", "kelas", "tingkat", "ukuran", "jumlah", "angka", "data", "informasi", "pengetahuan"
}

factory = StopWordRemoverFactory()
sastrawi_stop = set(factory.get_stop_words())
nltk_stop_id = set(stopwords.words("indonesian"))
nltk_stop_en = set(stopwords.words("english"))
combined_stopwords = sastrawi_stop.union(nltk_stop_id).union(nltk_stop_en).union(STOPWORDS_ID)

stemmer_factory_global = StemmerFactory()
indonesian_stemmer = stemmer_factory_global.create_stemmer()

ft_model = None
try:
    if os.path.exists(FASTTEXT_PATH):
        ft_model = fasttext.load_model(FASTTEXT_PATH)
    else:
        print(f"PERINGATAN: Model FastText tidak ditemukan di {FASTTEXT_PATH}")
except Exception as e:
    print(f"Gagal memuat FastText: {e}")

# =============================================================================
# FUNGSI-FUNGSI PIPELINE v3 (Sama persis dengan kode Anda)
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

def clean_rumus(text):
    text = re.sub(r'[𝑎-𝑧𝑨-𝒁𝟎-𝟗]', '', text)
    text = re.sub(r'^[0-9\.\,\=\-\+\*\/\(\) ]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s{0,3}[A-Za-z0-9]{1,3}\s*$', '', text, flags=re.MULTILINE)
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_pdf(pdf_path):
    header_even, header_odd, footer_all = [], [], []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages: return [], "", "PDF kosong."
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
                        if i % 2 == 0 and h in common_even: lines = lines[1:]
                        if i % 2 == 1 and h in common_odd: lines = lines[1:]
                        if f in common_footer: lines = lines[:-1]
                    raw = "\n".join(lines)
                    clean = clean_text(raw)
                    page_texts.append((i, clean))
                    all_text += " " + clean
        return page_texts, all_text.strip(), None
    except Exception as e:
        return [], "", f"Error membaca PDF: {e}"

def extract_chapters(pdf_path, max_pages_for_toc=15, target_chunks=8):
    def clean_line(line): return re.sub(r'\s+', ' ', line.strip())
    def is_heading(line):
        if len(line.split()) <= 10:
            if line.isupper() or re.match(r'^(BAB|Pasal|Bagian|Subbab)\s+[IVXLC\d]+', line) or re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$', line):
                return True
        return False
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages: pages_text.append(page.extract_text() or "")
    total_pages = len(pages_text)
    
    # 1. TOC
    toc_entries = []
    for i in range(min(max_pages_for_toc, total_pages)):
        lines = [clean_line(l) for l in pages_text[i].split("\n") if l.strip()]
        for line in lines:
            m = re.match(r'^(BAB\s+[IVXLC\d]+.*?)(?:\.{2,}|\s+)(\d+)', line)
            if m: toc_entries.append((m.group(1), int(m.group(2))))
    if toc_entries:
        chapters = []
        for idx, (title, start) in enumerate(toc_entries):
            end = toc_entries[idx + 1][1] if idx + 1 < len(toc_entries) else total_pages
            text = "\n".join(pages_text[start - 1: end])
            chapters.append((title, text))
        return chapters

    # 2. Heading
    headings = []
    for i, text in enumerate(pages_text, start=1):
        lines = [clean_line(l) for l in text.split("\n") if l.strip()]
        for line in lines:
            if is_heading(line): headings.append((line, i))
    if headings:
        chapters = []
        for idx, (title, start) in enumerate(headings):
            end = headings[idx + 1][1] if idx + 1 < len(headings) else total_pages
            text = "\n".join(pages_text[start - 1: end])
            chapters.append((title, text))
        return chapters

    # 3. Fallback
    pages_per_chunk = max(10, math.ceil(total_pages / target_chunks))
    chunks = []
    for i in range(0, total_pages, pages_per_chunk):
        start = i + 1
        end = min(i + pages_per_chunk, total_pages)
        text = "\n".join(pages_text[i:end])
        chunks.append((f"Halaman {start}-{end}", text))
    return chunks

def summarize_lexrank(text, sentence_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(s) for s in summary)

def build_final_combined_summary(book_title, chapters, sentence_count=5):
    chapter_summaries, combined_parts = [], []
    for title, content in chapters:
        cleaned = clean_rumus(content)
        summary = summarize_lexrank(cleaned, sentence_count=sentence_count).strip() + "."
        chapter_summaries.append(summary)
        combined_parts.append(f"{title}. {summary}")
    return f"{book_title}. " + " ".join(combined_parts), chapter_summaries

def clean_book_context(text):
    text = text.lower()
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-z\s\.]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_summary_keyword(text, combined_stopwords_set, top_per_n=350):
    keywords_all = []
    for n in [1, 2, 3]:
        kw_extractor = yake.KeywordExtractor(lan="multilingual", n=n, top=top_per_n)
        keywords_all.extend(kw_extractor.extract_keywords(text))
    seen = set()
    cleaned = []
    for kw, score in sorted(keywords_all, key=lambda x: x[1]):
        kw = kw.strip()
        if not kw: continue
        words = kw.split()
        if len(words) > 1 and len(set(words)) == 1: continue
        if any(w.lower() in combined_stopwords_set for w in words): continue
        if any(w.lower() in STOPWORDS_ID for w in words): continue
        norm = kw.lower()
        if norm not in seen:
            seen.add(norm)
            cleaned.append((kw, score))
        cleaned = sorted(cleaned, key=lambda x: x[1])
    return [kw for kw, _ in cleaned]

def filter_multiword_capitalized_phrases(keywords):
    filtered = []
    for kw, score in keywords:
        words = kw.split()
        if len(words) > 1:
            if any(w and w[0].isupper() for w in words): filtered.append((kw, score))
        else: filtered.append((kw, score))
    return filtered

def filter_phrases_with_existing_acronyms(keywords):
    single_upper = {kw.upper() for kw, _ in keywords if len(kw.split()) == 1 and kw.isupper()}
    filtered = []
    for kw, score in keywords:
        words = kw.split()
        if len(words) > 1:
            if all(w.isupper() for w in words):
                filtered.append((kw, score))
                continue
            upper_words = [w.upper() for w in words]
            if any(u in single_upper for u in upper_words): continue
        filtered.append((kw, score))
    return filtered

def normalize_repeated_words_second(phrases):
    cleaned = []
    for ph in phrases:
        words = ph.split()
        no_consec = []
        for w in words:
            if not no_consec or w != no_consec[-1]: no_consec.append(w)
        unique, seen = [], set()
        for w in no_consec:
            if w.lower() not in seen:
                seen.add(w.lower())
                unique.append(w)
        cleaned.append(" ".join(unique))
    return cleaned

def merge_reversed_phrases_second(phrases):
    normalized = {}
    for ph in phrases:
        key = tuple(sorted(ph.split()))
        if key not in normalized: normalized[key] = ph
    return list(normalized.values())

def is_base_word(word):
    w = word.lower().strip()
    if len(w) <= 3: return True
    return indonesian_stemmer.stem(w) == w

def filter_single_word_derivatives(keyphrases):
    return [ph for ph in keyphrases if len(ph.split()) > 1 or is_base_word(ph.split()[0])]

def run_yake_pipeline(all_text, combined_stopwords_set, top_per_n=450):
    keywords_all = []
    for n in [1, 2, 3]:
        kw_extractor = yake.KeywordExtractor(lan="multilingual", n=n, top=top_per_n)
        keywords_all.extend(kw_extractor.extract_keywords(all_text))
    seen = set()
    cleaned = []
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
    cleaned = filter_multiword_capitalized_phrases(cleaned)
    cleaned = filter_phrases_with_existing_acronyms(cleaned)
    keyphrases = [kw.lower() for kw, _ in cleaned]
    keyphrases = normalize_repeated_words_second(keyphrases)
    keyphrases = merge_reversed_phrases_second(keyphrases)
    keyphrases = filter_single_word_derivatives(keyphrases)
    return keyphrases, cleaned

def normalize_text(text): return re.sub(r'[^a-z0-9\s]', ' ', text.lower()).strip()

def phrase_embedding_fasttext(phrase, model):
    tokens = normalize_text(phrase).split()
    if not tokens: return None
    vecs = [model.get_word_vector(t) for t in tokens]
    return np.mean(vecs, axis=0)

def compare_keywords_fasttext(keyphrases, context_keywords_clean, ft_model, threshold=0.5):
    if ft_model is None:
        return [{'keyword_full': kw, 'best_context_match': None, 'score': 1.0, 'method': 'no-filter'} for kw in keyphrases]
    ctx_data = []
    for kw_ctx in context_keywords_clean:
        ctx_text = normalize_text(kw_ctx)
        if not ctx_text: continue
        vec = phrase_embedding_fasttext(ctx_text, ft_model)
        if vec is not None: ctx_data.append((ctx_text, vec))
    if not ctx_data:
        return [{'keyword_full': kw, 'best_context_match': None, 'score': 1.0, 'method': 'no-filter'} for kw in keyphrases]
    matched = []
    for kw_full in keyphrases:
        full_text = normalize_text(kw_full)
        if not full_text: continue
        vec_full = phrase_embedding_fasttext(full_text, ft_model)
        if vec_full is None: continue
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

def tokenize3(text): return set(re.findall(r"\w+", text.lower()))

def map_keywords_to_pages(matched_keywords, page_texts, overlap_threshold=0.8):
    matched_phrases = [item['keyword_full'] for item in matched_keywords]
    keyword_pages = defaultdict(list)
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
    for phrase in keyword_pages: keyword_pages[phrase].sort()
    return dict(keyword_pages)

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
    doc = fitz.open(gt_path)
    keywords = set()
    for page in doc:
        text = page.get_text("text")
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        merged_lines, buffer = [], ""
        def is_complete_entry(line): return bool(re.search(r'[·,]\s*[ivxIVX\d]', line))
        def is_new_entry(line): return bool(re.match(r'^[A-Z]', line))
        for line in lines:
            if re.match(r'^[A-Z]$', line):
                if buffer: merged_lines.append(buffer); buffer = ""
                continue
            if buffer:
                if is_new_entry(line) and is_complete_entry(buffer):
                    merged_lines.append(buffer); buffer = line
                else:
                    buffer += " " + line
                    if is_complete_entry(buffer): merged_lines.append(buffer); buffer = ""
            else:
                if is_complete_entry(line): merged_lines.append(line)
                else: buffer = line
        if buffer: merged_lines.append(buffer)
        for line in merged_lines:
            match = re.match(r'^([A-Za-z][A-Za-z\s\-\(\)]*?)\s*[·,]', line.strip())
            if match:
                kw = match.group(1).strip()
                if len(kw) > 1: keywords.add(kw)
    return preprocess_phrases(keywords)

def fuzzy_match_evaluation(generated, ground_truth, threshold=85):
    tp, fp, matched_gt = set(), set(), set()
    for gen in generated:
        best_match, best_score = None, 0
        for gt in ground_truth:
            if gt in matched_gt: continue
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
    if not tokens or model is None: return np.zeros(300)
    return np.mean([model.get_word_vector(t) for t in tokens], axis=0)

def evaluasi_indeks(gt_path, matched_keywords, ft_model=None, use_fuzzy=True):
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    generated = preprocess_phrases([item['keyword_full'] for item in matched_keywords])
    ground_truth = extract_keywords_from_index_file(gt_path)

    if use_fuzzy: tp, fp, fn = fuzzy_match_evaluation(generated, ground_truth)
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
        "false_negatives": sorted(fn)
    }

# =============================================================================
# MAIN BULK PIPELINE
# =============================================================================

if __name__ == "__main__":
    print(f"\nMemulai Auto Indexing Bulk (Versi 3)...")
    print(f"Dataset path: {BASE_PATH}")
    print(f"Output path:  {OUTPUT_DIR}\n")

    all_excel_rows = []
    folders = [f for f in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, f))]

    for folder in folders:
        folder_path = os.path.join(BASE_PATH, folder)
        print(f"\n>>> Processing Folder: {folder}")

        pdf_files = os.listdir(folder_path)
        isi_pdf = None
        indeks_pdf = None

        for file in pdf_files:
            if file.lower().endswith(".pdf"):
                if "indeks" in file.lower():
                    indeks_pdf = os.path.join(folder_path, file)
                else:
                    isi_pdf = os.path.join(folder_path, file)

        output_file = os.path.join(OUTPUT_DIR, f"{folder}.txt")

        try:
            if isi_pdf is None:
                raise Exception("File isi PDF tidak ditemukan.")

            # 1. Judul Buku
            book_title = pemetaan_judul.get(folder, folder)
            print(f"  -> Judul: {book_title}")

            # 2. Preprocess PDF
            print("  -> Ekstrak dan Preprocess teks PDF...")
            page_texts, all_text, error = preprocess_pdf(isi_pdf)
            if error: raise Exception(error)

            # 3. Ekstraksi Bab & Summarization (V3 logic)
            print("  -> Ekstraksi Bab & LexRank Summarization...")
            chapters = extract_chapters(isi_pdf)
            final_combined_summary, _ = build_final_combined_summary(book_title, chapters)
            
            context_clean = clean_book_context(final_combined_summary)
            context_keywords_clean = extract_summary_keyword(context_clean, combined_stopwords, top_per_n=250)

            # 4. YAKE Pipeline (V3 logic)
            print("  -> YAKE Pipeline...")
            keyphrases, _ = run_yake_pipeline(all_text, combined_stopwords, top_per_n=450)

            # 5. FastText comparison
            print("  -> FastText Filtering...")
            matched_keywords = compare_keywords_fasttext(keyphrases, context_keywords_clean, ft_model, threshold=0.5)

            # 6. Map to Pages
            keyword_pages = map_keywords_to_pages(matched_keywords, page_texts)

            # 7. Evaluasi (jika ada file indeks ground truth)
            f1, precision, recall = 0.0, 0.0, 0.0
            if indeks_pdf:
                print("  -> Evaluasi terhadap Ground Truth...")
                eval_results = evaluasi_indeks(indeks_pdf, matched_keywords, ft_model, use_fuzzy=True)
                f1 = eval_results["f1_score"]
                precision = eval_results["precision"]
                recall = eval_results["recall"]
            else:
                print("  -> Peringatan: PDF Indeks Ground Truth tidak ditemukan.")

            # Tulis Laporan TXT
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("STATUS: SUCCESS\n\n")
                f.write(f"Judul Buku : {book_title}\n")
                f.write(f"F1 Score   : {f1:.4f}\n")
                f.write(f"Precision  : {precision:.4f}\n")
                f.write(f"Recall     : {recall:.4f}\n\n")
                f.write("="*50 + "\n\n")

                for kw, pages in keyword_pages.items():
                    pages_str = ", ".join(map(str, pages))
                    f.write(f"{kw} | {pages_str}\n")
                    
                    all_excel_rows.append({
                        "Folder": folder,
                        "Judul Buku": book_title,
                        "Keyword": kw,
                        "Pages": pages_str,
                        "F1 Score": f1,
                        "Precision": precision,
                        "Recall": recall
                    })

            print(f"  -> Selesai. Hasil tersimpan di: {output_file}")

        except Exception as e:
            print(f"  -> Error: {e}")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("STATUS: ERROR\n\n")
                f.write(str(e))

    # =============================================================================
    # EXPORT KE EXCEL
    # =============================================================================
    if all_excel_rows:
        df = pd.DataFrame(all_excel_rows)
        excel_path = os.path.join(OUTPUT_DIR, "rekap_semua_buku_v3_indeks.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"\n=======================================================")
        print(f"Selesai seluruh batch! Rekap excel tersimpan di:\n{excel_path}")
        print(f"=======================================================")
    else:
        print("\nTidak ada data yang berhasil diproses untuk diekspor ke Excel.")