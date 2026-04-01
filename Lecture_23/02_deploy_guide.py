"""
Лекция 23: Гайд по деплою на облачные платформы
=================================================
Пошаговые инструкции для Railway, Render и Fly.io.
Этот файл — справочник, запускать не нужно.
"""

# ============================================================
# 1. RAILWAY (https://railway.app)
# ============================================================
#
# Railway — самый простой способ задеплоить Docker-контейнер.
# Бесплатный план: $5 кредитов/месяц, хватает для демо.
#
# ─────────────────────────────────────────
# Способ A: Деплой из GitHub (рекомендуется)
# ─────────────────────────────────────────
#
# 1. Зайдите на https://railway.app и войдите через GitHub
#
# 2. New Project → Deploy from GitHub Repo
#    Выберите ваш репозиторий
#
# 3. Railway автоматически найдёт Dockerfile
#    Если нет — укажите:
#      Root Directory: Lecture_23
#
# 4. Добавьте переменные окружения:
#    Variables → Add Variable:
#      OPENAI_API_KEY = sk-proj-xxx
#      PORT = 8000
#
# 5. Settings → Networking → Generate Domain
#    Получите URL: https://your-app.up.railway.app
#
# 6. Проверьте:
#    curl https://your-app.up.railway.app/health
#
# ─────────────────────────────────────────
# Способ B: Деплой через CLI
# ─────────────────────────────────────────
#
# npm install -g @railway/cli
# railway login
# cd Lecture_23
# railway init
# railway up
# railway variables set OPENAI_API_KEY=sk-xxx
# railway domain    # Получить URL

# ============================================================
# 2. RENDER (https://render.com)
# ============================================================
#
# Render — альтернатива Railway с бесплатным планом.
#
# ─────────────────────────────────────────
# Пошагово:
# ─────────────────────────────────────────
#
# 1. Зайдите на https://render.com → New → Web Service
#
# 2. Подключите GitHub-репозиторий
#
# 3. Настройки:
#    Name: ai-chat-api
#    Runtime: Docker
#    Docker Build Context: Lecture_23
#    Dockerfile Path: Lecture_23/Dockerfile
#
# 4. Environment Variables:
#    OPENAI_API_KEY = sk-proj-xxx
#    PORT = 8000
#
# 5. Deploy! URL: https://ai-chat-api.onrender.com
#
# ─────────────────────────────────────────
# render.yaml (Infrastructure as Code)
# ─────────────────────────────────────────

RENDER_YAML = """
# render.yaml — положите в корень репозитория
services:
  - type: web
    name: ai-chat-api
    runtime: docker
    dockerContext: ./Lecture_23
    dockerfilePath: ./Lecture_23/Dockerfile
    envVars:
      - key: OPENAI_API_KEY
        sync: false             # Ввести вручную в дашборде
      - key: PORT
        value: 8000
    healthCheckPath: /health
    plan: free
"""

# ============================================================
# 3. FLY.IO (https://fly.io)
# ============================================================
#
# Fly.io — для тех кто хочет контроль над регионами.
# Бесплатный план: 3 shared VMs.
#
# ─────────────────────────────────────────
# Пошагово:
# ─────────────────────────────────────────
#
# 1. Установите CLI:
#    curl -L https://fly.io/install.sh | sh
#    fly auth login
#
# 2. Инициализация:
#    cd Lecture_23
#    fly launch
#      → Выберите имя: ai-chat-api
#      → Выберите регион (ams = Амстердам)
#      → Dockerfile detected? Yes
#
# 3. Секреты:
#    fly secrets set OPENAI_API_KEY=sk-xxx
#
# 4. Деплой:
#    fly deploy
#
# 5. URL: https://ai-chat-api.fly.dev
#    fly open       # Открыть в браузере

FLY_TOML = """
# fly.toml — создаётся автоматически при fly launch
app = "ai-chat-api"
primary_region = "ams"

[build]
  dockerfile = "Dockerfile"

[env]
  PORT = "8000"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true

[[vm]]
  size = "shared-cpu-1x"
  memory = "256mb"
"""

# ============================================================
# 4. СРАВНЕНИЕ ПЛАТФОРМ
# ============================================================
#
# ┌─────────────┬──────────┬──────────┬──────────┐
# │             │ Railway  │ Render   │ Fly.io   │
# ├─────────────┼──────────┼──────────┼──────────┤
# │ Бесплатно   │ $5/мес   │ 750 ч/м │ 3 VM     │
# │ Docker      │ ✅       │ ✅       │ ✅       │
# │ Auto-deploy │ ✅       │ ✅       │ ✅       │
# │ Custom URL  │ ✅       │ ✅       │ ✅       │
# │ CLI         │ ✅       │ ❌       │ ✅       │
# │ Простота    │ ⭐⭐⭐   │ ⭐⭐⭐   │ ⭐⭐     │
# │ Регионы     │ US/EU    │ US/EU    │ 30+      │
# └─────────────┴──────────┴──────────┴──────────┘
#
# Рекомендация для начинающих: Railway
# Для production: Fly.io или Render

# ============================================================
# 5. ЧЕКЛИСТ ПЕРЕД ДЕПЛОЕМ
# ============================================================
#
# [ ] Dockerfile собирается: docker build -t app .
# [ ] Контейнер запускается: docker run -p 8000:8000 app
# [ ] Health check проходит: curl localhost:8000/health
# [ ] .dockerignore исключает .env, .git, __pycache__
# [ ] Секреты НЕ в коде (только через env vars)
# [ ] Приложение слушает 0.0.0.0 (не 127.0.0.1!)
# [ ] PORT берётся из переменной окружения
# [ ] Логи идут в stdout (docker logs работает)
# [ ] CORS настроен для вашего фронтенда

if __name__ == "__main__":
    print("=" * 60)
    print("  Лекция 23: Гайд по деплою")
    print("=" * 60)
    print()
    print("  Платформы:")
    print("    1. Railway  — https://railway.app (рекомендуем)")
    print("    2. Render   — https://render.com")
    print("    3. Fly.io   — https://fly.io")
    print()
    print("  Подробные инструкции — в комментариях этого файла.")
    print()
    print("  render.yaml:")
    print(RENDER_YAML)
    print()
    print("  fly.toml:")
    print(FLY_TOML)
