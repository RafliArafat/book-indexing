# Production Deployment Checklist untuk Railway

Gunakan checklist ini sebelum deploy ke production di Railway.

## Pre-Deployment Checklist

### Code & Repository
- [ ] Semua changes di-commit dan di-push ke main branch
- [ ] Repository public (jika menggunakan Railway free tier)
- [ ] `.gitignore` sudah sesuai
- [ ] Tidak ada hardcoded secrets dalam code
- [ ] `requirements.txt` up-to-date dengan semua dependencies

### Configuration
- [ ] `Dockerfile` sudah tested locally
- [ ] `railway.toml` sudah configured
- [ ] `wsgi.py` sudah ada dan correct
- [ ] `.env.example` lengkap dengan semua variables yang needed
- [ ] Model FastText path sudah verified

### Environment Variables (di Railway)
- [ ] `FLASK_SECRET_KEY` - Generate random key baru untuk production
  ```python
  python -c "import secrets; print(secrets.token_urlsafe(32))"
  ```
- [ ] `FLASK_ENV=production`
- [ ] `FASTTEXT_MODEL_PATH` - Set ke path yang correct
- [ ] Jika ada database, credentials sudah set

### Security
- [ ] ✅ `DEBUG=False` di production (sudah di wsgi.py)
- [ ] ✅ `app.secret_key` di-read dari environment
- [ ] ✅ Tidak ada sensitive data di logs
- [ ] ✅ Error handling sudah proper
- [ ] Upload file validation sudah implemented
- [ ] CSRF protection enabled untuk forms

### Performance
- [ ] Gunicorn workers: `--workers 2` (standard untuk Railway)
- [ ] Timeout: `--timeout 120` (cukup untuk PDF processing)
- [ ] Max upload size: 50MB (sesuai dengan `UPLOAD_FOLDER` limit)
- [ ] Static files serving sudah configured

### Monitoring & Logging
- [ ] Logging sudah configured untuk production
- [ ] Error logging sudah setup
- [ ] Railway health check endpoint accessible
- [ ] Logs dapat diakses via Railway Dashboard

## Deployment Steps di Railway

### Step 1: Create Railway Account & Project
- [ ] Sign up di [railway.app](https://railway.app)
- [ ] Create new project
- [ ] Connect GitHub account

### Step 2: Configure Railway Project
```yaml
# Di Railway Dashboard:
1. Click "+ New Service"
2. Select "Database" (jika needed) atau "GitHub"
3. Select repository
4. Railway auto-detect Dockerfile
5. Add environment variables
```

### Step 3: Environment Variables
Di Railway Dashboard → Variables:
```
FLASK_SECRET_KEY=<generated-key>
FLASK_ENV=production
FASTTEXT_MODEL_PATH=/app/models/cc.id.300.bin
```

### Step 4: Deploy
```
1. Railway akan auto-deploy saat push ke main branch
2. Atau manual deploy via Dashboard
3. Monitor logs di Railway Dashboard
```

## Post-Deployment Verification

### Immediate (5 minutes)
- [ ] Application started successfully (check logs)
- [ ] No error 500
- [ ] Health check passing
- [ ] Public URL accessible

### Functional (15 minutes)
- [ ] Home page loads
- [ ] File upload form works
- [ ] Basic functionality tested
- [ ] No console errors

### Performance (1 hour)
- [ ] Response time acceptable
- [ ] Memory usage stable
- [ ] No memory leaks
- [ ] CPU usage normal

### Complete (24 hours)
- [ ] No unexpected errors in logs
- [ ] All features working as expected
- [ ] Performance acceptable under load
- [ ] Database connections stable (jika ada)

## Common Issues & Solutions

### Build Fails
```
Issue: Docker build error
Solution:
1. Check Dockerfile syntax
2. Verify all files exist (requirements.txt, wsgi.py)
3. Test locally: docker build .
4. Check logs di Railway
```

### Application Crashes
```
Issue: App starts but crashes immediately
Solution:
1. Check environment variables set correctly
2. Check model file path
3. Look at logs for specific error
4. Test locally with same env vars
```

### Model Not Found
```
Issue: FastText model missing
Solution:
1. Verify path in FASTTEXT_MODEL_PATH
2. Upload model to /models directory
3. Or download during build (update Dockerfile)
4. Check file permissions
```

### Slow Performance
```
Issue: Application slow/unresponsive
Solution:
1. Increase workers: --workers 4 (if CPU allows)
2. Optimize PDF processing
3. Add caching
4. Monitor memory usage
```

## Monitoring in Production

### Via Railway Dashboard
- Go to project
- Click "Deployments" tab
- View:
  - Build logs
  - Runtime logs
  - Metrics (CPU, Memory, Network)
  - Status indicator

### Check Health
```bash
curl https://your-railway-url.railway.app/
```

### View Logs
```bash
# Via Railway CLI
railway logs

# Or via Dashboard → Logs tab
```

## Rollback Procedure

Jika ada masalah serius:

1. Di Railway Dashboard → Deployments
2. Pilih deployment sebelumnya yang berhasil
3. Click "Redeploy"
4. Atau rollback akan automatic jika ada failure

## Scaling (jika needed)

Railway's free tier sudah cukup untuk:
- Small to medium traffic
- Single instance
- Limited concurrent users

Untuk scale up:
1. Upgrade Railway plan
2. Increase resource allocation
3. Add database (managed PostgreSQL, MySQL, etc)
4. Use Railway environment management

## Backup & Disaster Recovery

- [ ] Regular git commits
- [ ] Database backups (if applicable)
- [ ] Model files backed up
- [ ] Configuration documented
- [ ] Recovery procedure tested

## Maintenance Schedule

- [ ] Weekly: Check logs for errors
- [ ] Monthly: Review performance metrics
- [ ] Quarterly: Update dependencies
- [ ] As needed: Apply security patches

## Support & Documentation

- Railway Docs: https://docs.railway.app
- Flask Docs: https://flask.palletsprojects.com
- Gunicorn Docs: https://gunicorn.org
- FastText Docs: https://fasttext.cc

---

**Last Updated**: 2024
**Maintained By**: [Your Name/Team]
**Next Review**: [Date]
