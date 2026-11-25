## Resumen

<!-- Describe de forma clara y concisa qué aporta esta Pull Request y qué parte del proyecto cubre. -->

## Tipo de cambio

- [ ] `feat`: Nueva funcionalidad
- [ ] `fix`: Corrección de error
- [ ] `docs`: Documentación y comentarios
- [ ] `refactor`: Refactorización sin cambio funcional
- [ ] `test`: Tests o benchmarks experimentales

## Cambios realizados por módulo

<!-- Lista los archivos modificados o creados y explica brevemente la responsabilidad de cada uno. -->

## Decisiones técnicas y arquitectura

<!-- Explica las decisiones tomadas (modelos, parámetros, degradación elegante, rendimiento, etc.). -->

## Plan de pruebas y verificación

```bash
# Comandos ejecutados tras el último cambio y resultado obtenido
```

## Checklist

- [ ] Se respeta el desacoplamiento estricto entre Capa 1 (`src/ui/`), Capa 2 (`src/core/`) y Capa 3 (`src/audio/`).
- [ ] Se respetan los contratos de datos (`src/core/schemas.py` y `src/config.py`) y la invariante PCM mono `16 000 Hz` `float32` en `[-1.0, 1.0]`.
- [ ] El código nuevo o modificado está documentado con *docstrings* y comentarios claros.
- [ ] Se han ejecutado y verificado los tests tras la última edición.
