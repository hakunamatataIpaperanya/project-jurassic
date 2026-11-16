#!/bin/sh
# เว็บแอปแผนที่ซากดึกดำบรรพ์ + โมเดลสด ของ Project_Jurassic -> http://127.0.0.1:8050  (หยุดด้วย Ctrl+C)
# ข้อมูลและโมเดลทั้งหมดอ่านจากโฟลเดอร์นี้เท่านั้น (data/, models/, notebooks/*_lib.py)
# ใช้ได้ทั้ง macOS และ Linux: macOS กันเครื่องหลับระหว่างรันด้วย caffeinate (Linux ไม่มี ให้ตั้งเครื่องไม่ให้หลับเอง)
# ตัวแปรที่ปรับได้: PORT (8050), PJ_MAX_PREDICT (คิวทำนายสด 4), PJ_MAX_INFLIGHT (คำขอพร้อมกัน 48)
cd "$(dirname "$0")/.." || exit 1
PORT="${PORT:-8050}"
if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "พอร์ต $PORT ถูกใช้อยู่แล้ว (อาจเป็นเว็บของโปรเจกต์อื่น):"
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | tail -n +2 | awk '{print "  PID " $2 ": " $1}'
  echo "เลือกพอร์ตอื่น เช่น  PORT=8051 sh scripts/run_webapp.sh"
  exit 1
fi
echo "Project_Jurassic web app -> http://127.0.0.1:$PORT"
if command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -i .venv/bin/python -m uvicorn webapp.server:app --host 127.0.0.1 --port "$PORT"
fi
exec .venv/bin/python -m uvicorn webapp.server:app --host 127.0.0.1 --port "$PORT"
