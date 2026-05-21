# Railway Deployment - Quick Start

File konfigurasi untuk deploy aplikasi di Railway telah siap!

## File yang dibuat:

1. **requirements.txt** - Dependencies Python
2. **Dockerfile** - Container configuration
3. **railway.toml** - Railway-specific configuration
4. **.env.example** - Environment variables template
5. **wsgi.py** - WSGI entry point untuk production
6. **docker-compose.yml** - Development setup
7. **.gitignore** - Git configuration
8. **railway-deploy.md** - Detailed deployment guide

## Quick Deploy Steps:

### 1. Push ke GitHub
```bash
git add .
git commit -m "Add Railway deployment files"
git push origin main
```

### 2. Di Railway Dashboard
- Klik "+ New Project"
- Select "Deploy from GitHub"
- Pilih repository ini
- Railway otomatis detect Dockerfile

### 3. Set Environment Variables di Railway
```
FLASK_SECRET_KEY=<random-key>
FLASK_ENV=production
```

### 4. Deploy!
Railway akan build dan deploy otomatis.

## Local Testing:

Test dengan Docker sebelum deploy:

```bash
docker build -t book-indexing .
docker run -p 5000:5000 -e FLASK_SECRET_KEY=dev-key book-indexing
```

Atau gunakan docker-compose:
```bash
docker-compose up
```

## Important Notes:

⚠️ **Model FastText**: 
- Model `cc.id.300.bin` harus tersedia
- Update path di environment variable jika needed
- Lihat `railway-deploy.md` untuk opsi upload

⚠️ **Secret Key**:
- Jangan hardcode di code
- Generate unique key untuk production
- Set via Railway environment variables

⚠️ **File Upload Limits**:
- Railway default 50MB
- Bisa disesuaikan via `MAX_CONTENT_LENGTH`

## Troubleshooting:

Lihat `railway-deploy.md` untuk detailed troubleshooting guide.

## Next Steps:

1. Read `railway-deploy.md` untuk full documentation
2. Push code ke GitHub
3. Deploy via Railway Dashboard
4. Monitor aplikasi di Railway

Good luck! 🚀
