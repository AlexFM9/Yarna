"""Módulo de entrada acústica: normalización PCM, buffer circular y Silero VAD adaptativo.

Este módulo implementa la primera etapa del pipeline de audio (Capa 3) de YARNA.
Su misión es transformar la señal cruda del micrófono en fragmentos de voz limpios,
estandarizados y listos para el reconocedor de voz (STT):

1. **Estandarización PCM (`normalize_pcm`)**:
   Todos los módulos internos operan en audio PCM mono a 16 000 Hz (`SAMPLE_RATE`)
   en coma flotante (`float32`) acotado en ``[-1.0, 1.0]``. Además de convertir
   muestras enteras (como ``int16``) o estéreo a mono, se elimina el desplazamiento
   continuo (*DC offset*) y se normaliza el pico de amplitud sin amplificar el
   silencio de fondo.

2. **Buffer Circular de Pre-roll (`CircularPCMBuffer`)**:
   Un detector de voz tarda unos milisegundos en confirmar que el alumno ha
   empezado a hablar. Si solo grabáramos desde el instante de detección,
   recortaríamos los primeros fonemas (por ejemplo, una "s" o una "p" inicial).
   El buffer circular retiene continuamente los últimos 0.5 segundos de audio
   para anteponerlos automáticamente cuando arranca la locución.

3. **Detección de Actividad Vocal (`VoiceActivityDetector`)**:
   Combina la red neuronal ligera **Silero VAD** (en ventanas nativas de 512
   muestras = 32 ms a 16 kHz) con una estimación adaptativa de ruido de fondo
   ($E_{base} + \\alpha \\cdot \\sigma$). Permite funcionar tanto en *streaming*
   trozo a trozo (`process_chunk`) como sobre grabaciones completas de un turno
   (`extract_speech_segment`), respetando la paciencia de silencio configurada por
   nivel MCER (2.0 s para `A2`, 0.9 s para `B2/C1`).
"""

from __future__ import annotations

from collections import deque
import logging
from typing import Any

import numpy as np

from src.config import LEVEL_PARAMS, SAMPLE_RATE

logger = logging.getLogger(__name__)

# Silero VAD v5 exige exactamente 512 muestras por inferencia cuando trabaja a 16 kHz
# (512 muestras / 16 000 Hz = 0.032 segundos = 32 ms por ventana).
SILERO_WINDOW_SAMPLES: int = 512


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Convierte cualquier array PCM entero o flotante a un vector 1D `float32` en `[-1.0, 1.0]`.

    A diferencia de `normalize_pcm`, esta función auxiliar NO realiza normalización de
    pico por bloque ni resta la media global, de modo que preserva el volumen relativo
    entre trozos consecutivos durante el streaming del micrófono.

    Args:
        audio: Array NumPy con muestras de audio (1D mono o 2D estéreo/multicanal,
            en formato entero como `int16`/`int32`/`uint8` o flotante `float32`/`float64`).

    Returns:
        Array unidimensional (`1D`) de tipo `np.float32` con valores acotados en `[-1.0, 1.0]`.
    """
    arr = np.asarray(audio)
    if arr.size == 0:
        return np.empty(0, dtype=np.float32)

    # Si el micrófono o fichero entrega múltiples dimensiones (por ejemplo, audio estéreo
    # en formato (N, 2) o (2, N)), primero eliminamos ejes unitarios y luego promediamos
    # los canales para obtener una única pista mono.
    if arr.ndim > 1:
        arr = np.squeeze(arr)
        if arr.ndim > 1:
            # Asumimos que el eje con menor tamaño corresponde al número de canales.
            channel_axis = 1 if arr.shape[1] <= arr.shape[0] else 0
            arr = np.mean(arr.astype(np.float64), axis=channel_axis)

    # Conversión de tipos enteros (PCM clásico de tarjeta de sonido/WAV) a [-1.0, 1.0].
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        if info.min == 0:
            # PCM sin signo (p. ej. uint8 de 0..255): el punto medio (128) equivale al silencio (0.0).
            midpoint = (float(info.max) + 1.0) / 2.0
            float_arr = (arr.astype(np.float32) - midpoint) / midpoint
        else:
            # PCM con signo (p. ej. int16 de -32768..32767): dividimos por la máxima magnitud.
            scale = float(max(abs(info.min), info.max))
            float_arr = arr.astype(np.float32) / scale
    else:
        # Si ya viene en coma flotante, limpiamos posibles valores NaN o infinitos.
        float_arr = np.nan_to_num(
            arr.astype(np.float32, copy=False),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

    # Garantizamos estrictamente la invariante de rango [-1.0, 1.0].
    return np.clip(float_arr, -1.0, 1.0).astype(np.float32, copy=False)


def normalize_pcm(
    audio: np.ndarray,
    noise_floor: float = 1e-4,
    peak_normalize: bool = True,
) -> np.ndarray:
    """Normaliza una señal PCM completa a `float32` en `[-1.0, 1.0]` con centrado DC seguro.

    El centrado DC resta la media de la señal para centrar la onda en 0.0 y eliminar
    desplazamientos eléctricos habituales en micrófonos económicos. Después, si la señal
    supera el umbral `noise_floor`, escala el pico máximo a 1.0 para compensar diferencias
    de ganancia entre distintos dispositivos.

    Args:
        audio: Señal de entrada en cualquier formato PCM soportado por `_to_mono_float32`.
        noise_floor: Amplitud mínima necesaria para aplicar ganancia de pico. Si el audio es
            prácticamente silencio (`peak < noise_floor`), se evita amplificar ruido de fondo.
        peak_normalize: Si es `True`, escala la onda para que su máximo absoluto sea 1.0.

    Returns:
        Array `1D` `np.float32` centrado y acotado en `[-1.0, 1.0]`.
    """
    float_arr = _to_mono_float32(audio)
    if float_arr.size == 0:
        return float_arr

    # Se calcula la media en float64 para evitar errores de acumulación numérica en audios largos.
    centered = float_arr - np.float32(np.mean(float_arr, dtype=np.float64))

    if peak_normalize:
        peak = float(np.max(np.abs(centered)))
        # Solo amplificamos si hay señal real por encima del piso de ruido electrónico.
        if peak >= noise_floor:
            centered = centered / np.float32(peak)

    return np.clip(centered, -1.0, 1.0).astype(np.float32, copy=False)


class CircularPCMBuffer:
    """Buffer circular deslizante que conserva el audio inmediatamente previo al habla.

    Guarda como máximo los últimos `max_seconds` segundos de audio. Al trabajar por bloques
    en una cola `deque`, añadir y recortar audio es muy rápido sin tener que copiar todo el
    array en cada muestra.
    """

    def __init__(self, max_seconds: float = 0.5, sample_rate: int = SAMPLE_RATE) -> None:
        """Inicializa el buffer circular.

        Args:
            max_seconds: Duración máxima de historia retenida (por defecto 0.5 s de pre-roll).
            sample_rate: Frecuencia de muestreo en Hz (por defecto 16 000 Hz).

        Raises:
            ValueError: Si `max_seconds` o `sample_rate` no son positivos.
        """
        if max_seconds <= 0:
            raise ValueError("max_seconds debe ser positivo.")
        if sample_rate <= 0:
            raise ValueError("sample_rate debe ser positivo.")
        self.max_seconds = max_seconds
        self.sample_rate = sample_rate
        self.max_samples = max(1, int(round(max_seconds * sample_rate)))
        self._chunks: deque[np.ndarray] = deque()
        self._total_samples: int = 0

    def append(self, samples: np.ndarray) -> None:
        """Añade un bloque de muestras al final del buffer y descarta el exceso más antiguo.

        Args:
            samples: Fragmento de audio entrante (cualquier formato compatible con `_to_mono_float32`).
        """
        chunk = _to_mono_float32(samples)
        if chunk.size == 0:
            return

        # Si el bloque nuevo por sí solo supera la capacidad del buffer, nos quedamos
        # únicamente con sus últimas `max_samples` muestras.
        if chunk.size >= self.max_samples:
            self._chunks.clear()
            self._chunks.append(chunk[-self.max_samples :].copy())
            self._total_samples = self.max_samples
            return

        self._chunks.append(chunk.copy())
        self._total_samples += chunk.size

        # Eliminamos muestras por la izquierda (lo más antiguo) hasta respetar el límite.
        while self._total_samples > self.max_samples and self._chunks:
            excess = self._total_samples - self.max_samples
            oldest = self._chunks[0]
            if oldest.size <= excess:
                self._chunks.popleft()
                self._total_samples -= oldest.size
            else:
                self._chunks[0] = oldest[excess:]
                self._total_samples -= excess

    def read_all(self) -> np.ndarray:
        """Concatena y devuelve todo el audio contenido actualmente en la ventana circular.

        Returns:
            Vector `1D` `np.float32` con el historial más reciente en orden cronológico.
        """
        if not self._chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(list(self._chunks)).astype(np.float32, copy=False)

    def clear(self) -> None:
        """Vacía por completo el buffer circular."""
        self._chunks.clear()
        self._total_samples = 0


class VoiceActivityDetector:
    """Detector de actividad vocal (VAD) dual basado en Silero VAD y umbral adaptativo.

    Soporta dos modos de uso para adaptarse a la interfaz Gradio:
    - **Modo streaming (`process_chunk`)**: recibe trozos continuos del micrófono,
      detecta cuándo empieza la voz (recuperando el pre-roll) y devuelve el turno entero
      cuando el alumno guarda silencio el tiempo configurado para su nivel (`A2` o `B2/C1`).
    - **Modo bloque (`extract_speech_segment`)**: recibe una grabación ya terminada y
      recorta los silencios innecesarios del principio y del final.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        speech_threshold: float = 0.5,
        alpha: float = 2.0,
        pre_roll_sec: float = 0.5,
        use_silero: bool = True,
    ) -> None:
        """Inicializa el detector de voz y los estadísticos adaptativos de ruido.

        Args:
            sample_rate: Frecuencia de muestreo en Hz (Silero VAD solo admite 8000 o 16000 Hz).
            speech_threshold: Probabilidad mínima [0, 1] de Silero para considerar que hay voz.
            alpha: Multiplicador de desviación típica en la fórmula $E_{base} + \\alpha \\cdot \\sigma$.
            pre_roll_sec: Segundos de audio previo conservados antes del inicio de voz.
            use_silero: Si es `True`, intenta cargar el modelo neuronal Silero VAD. Si no está
                instalado, degrada automáticamente a detección por energía RMS adaptativa.

        Raises:
            ValueError: Si `sample_rate` no es 8000 ni 16000 Hz.
        """
        if sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD requiere sample_rate de 8000 o 16000 Hz.")
        self.sample_rate = sample_rate
        self.speech_threshold = speech_threshold
        self.alpha = alpha
        # Silero usa ventanas de 512 muestras a 16 kHz (o 256 muestras a 8 kHz = 32 ms).
        self.window_samples = SILERO_WINDOW_SAMPLES if sample_rate == 16000 else 256

        # Estado de segmentación temporal para streaming.
        self._pre_roll = CircularPCMBuffer(max_seconds=pre_roll_sec, sample_rate=sample_rate)
        self._pending = np.empty(0, dtype=np.float32)  # Resto < 512 muestras entre llamadas.
        self._speech_frames: list[np.ndarray] = []
        self._is_speaking: bool = False
        self._silence_samples: int = 0
        self._speech_samples: int = 0

        # Estimación adaptativa del ruido ambiente (E_base = media, sigma = desviación típica).
        self._e_base: float = 0.005
        self._sigma: float = 0.002
        self._min_energy_floor: float = 0.01

        self._model: Any = None
        self._torch: Any = None
        if use_silero:
            self._load_silero_model()

    def _load_silero_model(self) -> None:
        """Intenta cargar Silero VAD en formato ONNX (más rápido) o TorchScript como respaldo."""
        try:
            import torch
            from silero_vad import load_silero_vad

            self._torch = torch
            try:
                self._model = load_silero_vad(onnx=True)
            except Exception:
                self._model = load_silero_vad(onnx=False)
        except Exception as exc:
            logger.warning(
                "Silero VAD no disponible (%s). Usando detector adaptativo por energía RMS.",
                exc,
            )
            self._model = None

    def _update_noise_stats(self, rms: float) -> None:
        """Actualiza suavemente el ruido de fondo durante los tramos sin voz (media móvil EMA)."""
        decay = 0.95
        diff = rms - self._e_base
        self._e_base = decay * self._e_base + (1.0 - decay) * rms
        var = decay * (self._sigma**2) + (1.0 - decay) * (diff**2)
        self._sigma = float(np.sqrt(max(var, 1e-10)))

    def _adaptive_energy_threshold(self) -> float:
        """Calcula el umbral dinámico de energía: $E_{base} + \\alpha \\cdot \\sigma$."""
        return max(self._min_energy_floor, self._e_base + self.alpha * self._sigma)

    def _is_window_speech(self, window: np.ndarray) -> bool:
        """Decide si una ventana de 32 ms contiene habla humana."""
        # Energía eficaz (RMS) de la ventana para evitar falsos positivos con silencio eléctrico.
        rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
        energy_thresh = self._adaptive_energy_threshold()

        if self._model is not None and self._torch is not None:
            with self._torch.no_grad():
                tensor = self._torch.from_numpy(window)
                prob = float(self._model(tensor, self.sample_rate).item())
            # Si Silero está muy seguro (> 0.75), aceptamos aunque la voz sea suave;
            # en zona intermedia exigimos también energía por encima de la mitad del piso de ruido.
            if prob >= max(0.75, self.speech_threshold + 0.15):
                return True
            return prob >= self.speech_threshold and rms >= max(1e-4, self._e_base * 0.5)

        # Modo respaldo sin Silero instalado: comparación directa frente al umbral adaptativo.
        return rms >= energy_thresh

    def _reset_silero_states(self) -> None:
        """Reinicia la memoria oculta recurrente interna de Silero VAD entre locuciones."""
        if self._model is not None and hasattr(self._model, "reset_states"):
            self._model.reset_states()

    def reset(self) -> None:
        """Reinicia todo el estado de captura entre turnos o sesiones de conversación."""
        self._pre_roll.clear()
        self._pending = np.empty(0, dtype=np.float32)
        self._speech_frames.clear()
        self._is_speaking = False
        self._silence_samples = 0
        self._speech_samples = 0
        self._reset_silero_states()

    def _finalize_utterance(self, tail_keep_sec: float = 0.25) -> np.ndarray:
        """Ensambla la frase capturada, recorta el silencio sobrante final y normaliza el pico."""
        full = np.concatenate(self._speech_frames)
        # Conservamos hasta 0.25 s de cola tras la última palabra para no cortar consonantes finales,
        # eliminando el resto del tiempo de espera (por ejemplo, 1.75 s sobrantes en nivel A2).
        tail_keep_samples = int(round(tail_keep_sec * self.sample_rate))
        excess_silence = max(0, self._silence_samples - tail_keep_samples)
        if excess_silence > 0 and full.size > excess_silence:
            full = full[:-excess_silence]

        # Limpiamos buffers de turno (pero mantenemos _e_base y _sigma aprendidos del ambiente).
        self._speech_frames.clear()
        self._is_speaking = False
        self._silence_samples = 0
        self._speech_samples = 0
        self._pre_roll.clear()
        self._reset_silero_states()
        return normalize_pcm(full, peak_normalize=True)

    def process_chunk(self, audio_chunk: np.ndarray, level: str = "A2") -> np.ndarray | None:
        """Procesa un fragmento PCM incremental y devuelve la locución completa al terminar el turno.

        Args:
            audio_chunk: Trozo de audio entrante desde el stream del micrófono.
            level: Nivel MCER activo (`"A2"` o `"B2/C1"`). Determina cuánto silencio seguido
                se espera antes de dar el turno por finalizado (`LEVEL_PARAMS[level]["vad_silence_sec"]`).

        Returns:
            Array `float32` normalizado con la locución completa si se detectó fin de turno,
            o `None` mientras el alumno siga en silencio o hablando.
        """
        params = LEVEL_PARAMS.get(level, LEVEL_PARAMS["A2"])
        silence_limit_samples = int(round(float(params["vad_silence_sec"]) * self.sample_rate))

        # Convertimos el chunk a float32 sin normalizar pico por trozo para no distorsionar la energía.
        chunk = _to_mono_float32(audio_chunk)
        if chunk.size == 0:
            return None

        # Si quedaron muestras pendientes (< 512) de la llamada anterior, las anteponemos ahora.
        if self._pending.size > 0:
            chunk = np.concatenate([self._pending, chunk])

        # Dividimos en ventanas completas de 512 muestras; el sobrante queda guardado en _pending.
        total_windows = chunk.size // self.window_samples
        consumed = total_windows * self.window_samples
        self._pending = chunk[consumed:].copy()

        completed_utterance: np.ndarray | None = None

        for idx in range(total_windows):
            start = idx * self.window_samples
            window = chunk[start : start + self.window_samples]
            rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
            is_speech = self._is_window_speech(window)

            if not self._is_speaking:
                if is_speech:
                    # Arranque de voz: recuperamos el buffer circular para no perder la primera sílaba.
                    pre_roll_audio = self._pre_roll.read_all()
                    if pre_roll_audio.size > 0:
                        self._speech_frames.append(pre_roll_audio)
                    self._speech_frames.append(window.copy())
                    self._pre_roll.clear()
                    self._is_speaking = True
                    self._silence_samples = 0
                    self._speech_samples = window.size
                else:
                    # Silencio previo al habla: alimentamos el pre-roll y calibramos ruido ambiente.
                    self._pre_roll.append(window)
                    self._update_noise_stats(rms)
            else:
                self._speech_frames.append(window.copy())
                if is_speech:
                    # El alumno sigue hablando: reseteamos el contador de silencio.
                    self._silence_samples = 0
                    self._speech_samples += window.size
                else:
                    # Pausa dentro o al final de la frase: acumulamos silencio hasta alcanzar el límite.
                    self._silence_samples += window.size
                    if self._silence_samples >= silence_limit_samples:
                        completed_utterance = self._finalize_utterance()
                        break

        return completed_utterance

    def flush(self) -> np.ndarray | None:
        """Fuerza la entrega del audio acumulado si el usuario detiene el micrófono manualmente.

        Returns:
            Locución normalizada si había habla activa en curso, o `None` en caso contrario.
        """
        if not self._is_speaking or not self._speech_frames:
            return None
        return self._finalize_utterance()

    def extract_speech_segment(
        self,
        audio: np.ndarray,
        level: str = "A2",
        padding_sec: float = 0.25,
    ) -> np.ndarray:
        """Modo bloque completo: recorta silencios inicial y final de una grabación ya cerrada.

        Pensado para componentes UI que envían el audio completo al soltar el botón de grabación.

        Args:
            audio: Grabación completa de un turno.
            level: Nivel MCER (mantenido por consistencia de interfaz).
            padding_sec: Margen de seguridad (en segundos) conservado antes y después de la voz.

        Returns:
            Segmento de voz recortado y normalizado en `float32`, o array vacío si no hubo voz.
        """
        signal = _to_mono_float32(audio)
        if signal.size == 0:
            return signal

        self._reset_silero_states()
        speech_indices: list[int] = []
        total_windows = signal.size // self.window_samples

        # Recorremos toda la grabación anotando los índices de las ventanas que contienen voz.
        for idx in range(total_windows):
            start = idx * self.window_samples
            window = signal[start : start + self.window_samples]
            if self._is_window_speech(window):
                speech_indices.append(idx)

        self._reset_silero_states()

        if not speech_indices:
            return np.empty(0, dtype=np.float32)

        # Recortamos desde la primera hasta la última ventana con voz añadiendo el margen padding_sec.
        pad_samples = int(round(padding_sec * self.sample_rate))
        start_sample = max(0, speech_indices[0] * self.window_samples - pad_samples)
        end_sample = min(
            signal.size,
            (speech_indices[-1] + 1) * self.window_samples + pad_samples,
        )
        _ = LEVEL_PARAMS.get(level, LEVEL_PARAMS["A2"])
        return normalize_pcm(signal[start_sample:end_sample], peak_normalize=True)
