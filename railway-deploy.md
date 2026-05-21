# Deploy ke Railway

Panduan langkah demi langkah untuk deploy aplikasi ke Railway.

## Prasyarat

1. Akun [Railway.app](https://railway.app)
2. Git dan Git CLI terinstal
3. Railway CLI terinstal (opsional tapi recommended)

## Langkah 1: Persiapan Repository

```bash
cd book-indexing

# Inisialisasi git jika belum ada
git init

# Add files
git add .
git commit -m "Initial commit for Railway deployment"
```

## Langkah 2: Upload ke GitHub/GitLab

```bash
# Buat repository di GitHub
# Kemudian push code

git remote add origin https://github.com/username/repository.git
git branch -M main
git push -u origin main
```

## Langkah 3: Deploy di Railway

### Opsi A: Menggunakan Railway Dashboard (Recommended)

1. Kunjungi [Railway.app](https://railway.app)
2. Login dengan akun Anda
3. Klik "+ New Project"
4. Pilih "Deploy from GitHub"
5. Pilih repository Anda
6. Railway akan otomatis mendeteksi `Dockerfile` dan `railway.toml`
7. Konfigurasi environment variables (lihat langkah 4)
8. Deploy!

### Opsi B: Menggunakan Railway CLI

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Inisialisasi di project directory
railway init

# Deploy
railway up
```

## Langkah 4: Konfigurasi Environment Variables

Di Railway Dashboard, tambahkan environment variables berikut:

```
FLASK_SECRET_KEY=<generate-random-key>
FLASK_ENV=production
FASTTEXT_MODEL_PATH=/app/models/cc.id.300.bin
```

**Untuk generate secret key**, gunakan:
```python
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Langkah 5: Upload Model FastText (Important!)

Model FastText harus di-upload ke Railway. Pilih satu dari opsi berikut:

### Opsi A: Include di Repository (Tidak Recommended untuk file besar)
- Jika file `cc.id.300.bin` < 100MB, bisa langsung include di repo
- Update `models/` folder dengan model file

### Opsi B: Upload saat Runtime (Recommended)
- Buat script untuk download model saat container start
- Edit `Dockerfile` dan tambahkan:

```dockerfile
RUN python -c "import fasttext; fasttext.util.download_model('id')" \
    && mkdir -p models \
    && mv cc.id.300.bin models/
```

### Opsi C: Gunakan Railway File Storage
- Gunakan Railway plugin untuk file storage
- Mount ke `/app/models`

## Langkah 6: Verifikasi Deployment

Setelah deployment selesai:

1. Railway akan memberikan public URL
2. Test aplikasi dengan mengakses URL tersebut
3. Cek logs di Railway Dashboard untuk debugging

## Troubleshooting

### Build Fails
- Pastikan `requirements.txt` ada dan benar
- Pastikan `Dockerfile` valid
- Cek logs di Railway Dashboard

### Application Crashes
- Pastikan environment variables sudah diset
- Cek bahwa model FastText tersedia
- Lihat logs untuk error messages

### Model Not Found
- Pastikan path model sudah benar
- Verify bahwa model file sudah di-upload

### Port Issues
- Railway otomatis assign port via `$PORT` variable
- `Dockerfile` dan `railway.toml` sudah handle ini

## Performance Tips

1. **Gunicorn Workers**: Sesuaikan `--workers` di Dockerfile berdasarkan:
   - CPU cores: `workers = 2 * CPU + 1`
   - Railway default: 2 workers (cukup untuk development)

2. **Timeout**: Ekstraksi PDF besar mungkin membutuhkan timeout lebih panjang
   - Saat ini: 120 seconds
   - Bisa disesuaikan di Dockerfile

3. **Memory**: Monitor penggunaan memory di Railway Dashboard
   - FastText model bisa menggunakan memory signifikan
   - Pertimbangkan upgrade plan jika needed

## Monitoring

Railway menyediakan:
- Real-time logs
- Resource usage metrics
- Deployment history
- Error tracking

Akses semuanya dari Railway Dashboard.

## Rollback

Jika ada masalah:
1. Railway Dashboard → Deployments
2. Pilih deployment sebelumnya
3. Klik "Redeploy" atau "Rollback"

## Update Aplikasi

```bash
# Make changes locally
git add .
git commit -m "Update message"
git push origin main

# Railway akan otomatis redeploy
```

## Dokumentasi Lanjutan

- [Railway Docs](https://docs.railway.app)
- [Railway Environment Variables](https://docs.railway.app/develop/variables)
- [Railway Dockerfile](https://docs.railway.app/deploy/dockerfile)
