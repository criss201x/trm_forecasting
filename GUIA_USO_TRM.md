# Guía de Uso: Experimento TRM Series Temporales

##  Inicio Rápido

### Paso 1: Preparar datos
```python
# rchivo CSV debe tener columnas: 'fecha' y 'trm'
# Formato fecha: YYYY-MM-DD
```

### Paso 2: Ejecutar experimento base
```python
from trm_experiment import ejecutar_experimento

# Con datos reales
resultados = ejecutar_experimento("trm_to_rn.csv")

# O prueba con datos sintéticos primero
resultados = ejecutar_experimento()
```

### Paso 3: Modificar configuración
```python
# En trm_experiment.py, ajusta CONFIG:
CONFIG = {
    'fecha_inicio': '2023-01-01',      # Ajusta según tus datos
    'fecha_fin': '2024-11-22',
    'fecha_inicio_val': '2024-07-01',   # ~3 meses validación
    'fecha_inicio_test': '2024-10-01',  # ~2 meses test
    'transformacion': 'retorno_log',    # o primera diferencia
    'verbose': True
}
```

## Interpretación de Resultados

### 1. **Si ARIMA elige órdenes bajos (0,1,1) o (1,1,0)**
```
ARIMA(0,1,0) = Random Walk puro
→ La TRM es casi completamente aleatoria
→ Difícil mejorar predicción naive

ARIMA(0,1,1) = Suavizamiento exponencial simple
→ Hay algo de autocorrelación en errores
→ Pequeña mejora posible sobre random walk

ARIMA(1,1,0) = AR(1) con diferenciación
→ Existe dependencia del valor anterior
→ Potencial para mejora con modelos más complejos
```

### 2. **Métricas Clave a Observar**

#### MSE en niveles vs transformado
- **MSE transformado bajo, MSE niveles alto**: 
  - El modelo captura bien los cambios porcentuales
  - Pero errores se amplifican al convertir a pesos
  - Considera optimizar directamente en niveles

#### Dirección Correcta %
- **> 55%**: Hay estructura predictible
- **50-55%**: Casi aleatorio
- **< 50%**: El modelo está sesgado (revisar datos)

#### Error Máximo y Percentil 95
- **Error máximo >> Percentil 95**: 
  - Hay eventos extremos que el modelo no captura
  - Considera modelos de volatilidad (GARCH)

## Experimentos Sugeridos

### Experimento 1: Comparar Transformaciones
```python
# Con ambas transformaciones
for trans in ['diferencia', 'retorno_log']:
    CONFIG['transformacion'] = trans
    resultados = ejecutar_experimento()
    print(f"\n{trans}: MSE={resultados['ARIMA']['metricas']['nivel_mse']}")
```

### Experimento 2: Análisis de Ventanas Temporales
```python
# Diferentes períodos de entrenamiento
feDcio = ['2022-01-01', '2023-01-01', '2023-06-01']
for fecha in fechas_inicio:
    CONFIG['fecha_inicio'] = fecha
    resultados = ejecutar_experimento()
    # Analizar cómo cambia el performance
```

### Experimento 3: Validación de Estabilidad
```python
# ¿El modelo es consistente en diferentes períodos?
# Divide en 2 partes y compara MSE
```

## Agregar LSTM e Híbridos

### Paso 1: Importar extensión
```python
from trm_experiment import *
from trm_lstm_extension import LSTMModel, ModeloHibridoARIMALSTM

# Datos preparados
df = cargar_datos_trm("datos_trm.csv")
train, val, test = dividir_datos(df, '2024-07-01', '2024-10-01')
```

### Paso 2: Entrenar LSTM con Grid Search
```python
from trm_lstm_extension import grid_search_lstm

# Definir grid de parámetros
param_grid = {
    'ventana': [5, 10, 20],
    'neuronas': [[50], [50, 30], [100, 50]],
    'dropout': [0.1, 0.2],
    'epochs': [50],
    'batch_size': [32]
}

# Buscar mejores parámetros
serie_trans, _ = preparar_transformaciones(train, 'retorno_log')
serie_val_trans, _ = preparar_transformaciones(val, 'retorno_log')

best_params, all_results = grid_search_lstm(
    serie_trans, 
    serie_val_trans,
    param_grid
)
```

### Paso 3: Modelo Híbrido
```python
# Crear modelo híbrido con mejores parámetros
hibrido = ModeloHibridoARIMALSTM(
    arima_params={'max_p': 3, 'max_q': 3},
    lstm_params=best_params,
    verbose=True
)

# Entrenar
hibrido.fit(serie_trans)

# Backtest expansivo
metricas_hibrido, pred_hibrido, real_hibrido = backtest_expansivo(
    ModeloHibridoARIMALSTM,
    df['trm'],
    '2024-10-01',
    transformacion='retorno_log'
)
```

## Análisis de Resultados Esperados

### Escenario A: Mercado Eficiente
- Random Walk ≈ ARIMA ≈ LSTM
- Dirección correcta ~50%
- **Acción**: Enfocarse en gestión de riesgo, no predicción

### Escenario B: Estructura Lineal
- ARIMA > Random Walk
- LSTM ≈ ARIMA
- **Acción**: ARIMA suficiente, optimizar órdenes

### Escenario C: Estructura No Lineal
- LSTM > ARIMA > Random Walk
- Híbrido > LSTM
- **Acción**: Invertir en modelos complejos y features

## Checklist de Validación

Antes de confiar en el modelo, verificar:

- [ ] ¿El modelo supera consistentemente al Random Walk?
- [ ] ¿El performance es estable en diferentes períodos?
- [ ] ¿Los errores están distribuidos normalmente?
- [ ] ¿No hay patrones en los residuos?
- [ ] ¿El modelo generaliza bien a datos no vistos?

##  Tips Avanzados

### 1. Detectar Cambios de Régimen
```python
# Calcular volatilidad móvil
volatilidad_20d = df['trm'].pct_change().rolling(20).std()

# Si volatilidad actual > percentil 80 histórico
# → Dar más peso a modelos entrenados en períodos volátiles
```

### 2. Features Adicionales para LSTM
```python
# Agregar indicadores técnicos
df['sma_10'] = df['trm'].rolling(10).mean()
df['rsi'] = calculate_rsi(df['trm'])
df['volatilidad'] = df['trm'].pct_change().rolling(20).std()

# Usar múltiples features en LSTM
features = ['retorno', 'volatilidad', 'rsi']
```

### 3. Ensemble Adaptativo
```python
# Ponderar modelos según performance reciente
def calcular_pesos_adaptativos(errores_recientes):
    # Invertir errores (menor error = mayor peso)
    pesos = 1 / (errores_recientes + 0.001)
    return pesos / pesos.sum()
```

## Preguntas para Reflexión


1. **¿Por qué mi mejor modelo tiene este MSE específico?**
   - ¿Es ruido irreducible o falta información?

2. **¿Qué eventos causan los errores más grandes?**
   - ¿Puedo detectar estos eventos anticipadamente?

3. **¿Mi modelo está aprendiendo o memorizando?**
   - Compara train vs test performance

4. **¿Qué información externa mejoraría las predicciones?**
   - Tasas Fed, precio petróleo, eventos políticos

5. **¿Es el MSE la métrica correcta para mi objetivo?**
   - Considera costo asimétrico de errores

## Troubleshooting

### Problema: "ARIMA tarda mucho"
```python
# Reduce espacio de búsqueda
auto_arima(serie, max_p=3, max_q=3, stepwise=True)
```

### Problema: "LSTM sobreajusta"
```python
# Aumenta dropout y reduce complejidad
lstm_params = {
    'neuronas': [30],  # Menos neuronas
    'dropout': 0.3,    # Más dropout
    'patience': 5      # Early stopping más agresivo
}
```

### Problema: "MSE en niveles muy alto"
```python
# Opción 1: Optimizar directamente en niveles
# En lugar de transformar, predice directamente TRM

# Opción 2: Usar transformación Box-Cox
from scipy.stats import boxcox
trm_boxcox, lambda_param = boxcox(df['trm'])
```

## Referencias y recursos adicionales

- **Teoría**: Hyndman & Athanasopoulos - "Forecasting: Principles and Practice"
- **ARIMA**: Box, Jenkins & Reinsel - "Time Series Analysis"
- **LSTM**: Brownlee - "Deep Learning for Time Series Forecasting"
- **Finanzas**: Tsay - "Analysis of Financial Time Series"

---

**Recuerda**: No hay free lunch. Un modelo que funciona perfectamente en un régimen puede fallar en otro. La clave es entender **por qué** funciona cuando funciona.
