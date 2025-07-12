import subprocess
import os
import time

# مسار البيئة الافتراضية وملفات الخدمات
venv_path = r"C:\Users\lenovo\Desktop\IR\ir_v3\venv\Scripts\activate"
services = [
    ("Frontend Service", r"src\online_servics\frontend_service.py", 8000),
    ("Query Processing Service", r"src\online_servics\query_processing_service.py", 8003),
    ("TF-IDF Ranking Service", r"src\online_servics\ranking_tfidf_service.py", 8006),
    ("Embedding Ranking Service", r"src\online_servics\ranking_embedding_service.py", 8007),
    ("Hybrid Ranking Service", r"src\online_servics\ranking_hybrid_service.py", 8008),
    ("Vector Store Service", r"src\online_servics\vector_store_service.py", 8005)
]

def start_service(service_name, script_path, port):
    # أمر لتفعيل البيئة الافتراضية وتشغيل الخدمة
    activate_cmd = f'cmd.exe /k "{venv_path} && python {script_path} && echo Service {service_name} on port {port} started. Press Ctrl+C to stop."'
    process = subprocess.Popen(activate_cmd, shell=True)
    print(f"Started {service_name} on port {port} in a new terminal (PID: {process.pid})")
    return process

# بدء جميع الخدمات
processes = []
for service_name, script_path, port in services:
    process = start_service(service_name, script_path, port)
    processes.append(process)
    time.sleep(2)  # انتظار قصير لمنع تداخل التشغيل

# الانتظار حتى يتم إغلاق جميع العمليات يدويًا
try:
    for process in processes:
        process.wait()
except KeyboardInterrupt:
    print("Shutting down all services...")
    for process in processes:
        process.terminate()
    for process in processes:
        process.wait()




