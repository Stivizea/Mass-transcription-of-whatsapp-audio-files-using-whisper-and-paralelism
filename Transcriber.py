import os
import multiprocessing
import time
import whisper
import torch
from tqdm import tqdm

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
# Número de procesos de cómputo pesado. Para tu 4060 Ti de 8GB, 2 es el número ideal.
NUMERO_DE_TRABAJADORES_GPU = 2

# --- Rutas (se calculan automáticamente) ---
try:
    RUTA_BASE_SCRIPT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    RUTA_BASE_SCRIPT = os.path.abspath(".")

CARPETA_AUDIOS = os.path.join(RUTA_BASE_SCRIPT, "audios")
CARPETA_TRANSCRIPCIONES = os.path.join(RUTA_BASE_SCRIPT, "transcripciones")

# --- Configuraciones del Modelo ---
MODELO_WHISPER = "medium"
IDIOMA_AUDIO = "es"

# ==============================================================================
# Variable global para los procesos hijos
modelo_global = None

# --- FUNCIÓN QUE FALTABA ---
def crear_carpetas_necesarias():
    """Asegura que la carpeta de transcripciones exista antes de empezar."""
    if not os.path.exists(CARPETA_TRANSCRIPCIONES):
        print(f"📁 Creando carpeta de salida: {CARPETA_TRANSCRIPCIONES}")
        os.makedirs(CARPETA_TRANSCRIPCIONES)

def inicializar_trabajador():
    """
    Esta función se ejecuta UNA VEZ por cada proceso trabajador.
    Carga el modelo de Whisper en la memoria de este proceso.
    """
    global modelo_global
    print(f"Proceso {os.getpid()}: Cargando modelo de Whisper en la GPU...")
    modelo_global = whisper.load_model(MODELO_WHISPER)
    print(f"Proceso {os.getpid()}: Modelo cargado y listo.")

def procesar_archivo(ruta_audio):
    """
    Esta es la función que ejecuta cada trabajador para un solo archivo.
    Usa el modelo que ya está cargado en su memoria.
    """
    nombre_archivo = os.path.basename(ruta_audio)
    ruta_salida_txt = os.path.join(CARPETA_TRANSCRIPCIONES, f"{os.path.splitext(nombre_archivo)[0]}.txt")
    
    try:
        resultado = modelo_global.transcribe(ruta_audio, fp16=True, language=IDIOMA_AUDIO)
        with open(ruta_salida_txt, "w", encoding="utf-8") as f:
            f.write(resultado["text"])
        return (nombre_archivo, "ÉXITO", "")
    except Exception as e:
        # Si un archivo falla aquí, es probable que esté realmente corrupto.
        return (nombre_archivo, "FALLO", str(e))

def encontrar_archivos_pendientes():
    """Encuentra archivos que aún no tienen una transcripción y devuelve una lista."""
    print("Buscando archivos pendientes...")
    archivos_pendientes = []
    if not os.path.isdir(CARPETA_AUDIOS):
        return None # Devuelve None si la carpeta de audios no existe

    for nombre_archivo in sorted(os.listdir(CARPETA_AUDIOS)):
        if nombre_archivo.endswith(".opus"):
            ruta_salida_txt = os.path.join(CARPETA_TRANSCRIPCIONES, f"{os.path.splitext(nombre_archivo)[0]}.txt")
            if not os.path.exists(ruta_salida_txt):
                archivos_pendientes.append(os.path.join(CARPETA_AUDIOS, nombre_archivo))
    print(f"Se encontraron {len(archivos_pendientes)} archivos para procesar.")
    return archivos_pendientes

def main():
    if not torch.cuda.is_available():
        print("🔥 Error: CUDA no está disponible.")
        return

    crear_carpetas_necesarias() # Esta línea ahora funcionará correctamente
    archivos_a_procesar = encontrar_archivos_pendientes()

    if archivos_a_procesar is None:
        print(f"🔥 Error: No se pudo encontrar la carpeta de audios en la ruta: {CARPETA_AUDIOS}")
        return

    if not archivos_a_procesar:
        print("🎉 ¡No hay archivos pendientes! Todo el trabajo está hecho.")
        return

    print(f"🚀 Iniciando pool con {NUMERO_DE_TRABAJADORES_GPU} trabajadores...")

    # Creamos el pool de procesos. 'initializer' llama a nuestra función para cargar el modelo.
    with multiprocessing.Pool(processes=NUMERO_DE_TRABAJADORES_GPU, initializer=inicializar_trabajador) as pool:
        
        resultados = []
        # 'imap_unordered' es eficiente y nos da los resultados a medida que terminan
        with tqdm(total=len(archivos_a_procesar), desc="Transcripciones") as progress_bar:
            for resultado in pool.imap_unordered(procesar_archivo, archivos_a_procesar):
                if "FALLO" in resultado[1]:
                    print(f"\n[ERROR] Archivo: {resultado[0]} | Causa: {resultado[2][:100]}...")
                resultados.append(resultado)
                progress_bar.update(1)

    # Reporte final
    exitos = [r for r in resultados if r[1] == "ÉXITO"]
    fallos = [r for r in resultados if r[1] == "FALLO"]

    print("\n" + "="*50)
    print("🎉 ¡Proceso completado!")
    print(f"✅ Transcritos con éxito: {len(exitos)}")
    print(f"❌ Fallos: {len(fallos)}")
    
    if fallos:
        print("\n--- Archivos que fallaron ---")
        for ruta, _, error_msg in fallos:
            # Imprime solo los primeros 100 caracteres del error para no saturar la consola
            print(f"Archivo: {os.path.basename(ruta)} | Error: {error_msg[:100]}...") 
    
    print("="*50)

if __name__ == "__main__":
    # Es OBLIGATORIO en Windows proteger la ejecución principal de esta manera.
    multiprocessing.set_start_method("spawn", force=True)
    main()
